# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import logging
import typing as t

import bentoml

import openllm

if t.TYPE_CHECKING:
    import torch
    import transformers
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")
    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")

logger = logging.getLogger(__name__)

FIM_PREFIX = "<fim-prefix>"
FIM_MIDDLE = "<fim-middle>"
FIM_SUFFIX = "<fim-suffix>"
FIM_PAD = "<fim-pad>"
EOD = "<|endoftext|>"
FIM_INDICATOR = "<FILL_HERE>"


class StarCoder(openllm.LLM["transformers.GPTBigCodeForCausalLM", "transformers.GPT2TokenizerFast"]):
    __openllm_internal__ = True

    def llm_post_init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def import_kwargs(self):
        model_kwds = {
            "device_map": "auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None,
            "load_in_8bit": True if torch.cuda.device_count() > 1 else False,
            "torch_dtype": torch.float16,
        }
        tokenizer_kwds = {"padding_side": "left"}
        return model_kwds, tokenizer_kwds

    def import_model(
        self,
        model_id: str,
        tag: bentoml.Tag,
        *model_args: t.Any,
        tokenizer_kwds: dict[str, t.Any],
        **attrs: t.Any,
    ) -> bentoml.Model:
        torch_dtype = attrs.pop("torch_dtype", torch.float16)
        load_in_8bit = attrs.pop("load_in_8bit", True)
        device_map = attrs.pop("device_map", "auto")

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **tokenizer_kwds)
        tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD],
                "pad_token": EOD,
            }
        )

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit, device_map=device_map, **attrs
        )
        try:
            return bentoml.transformers.save_model(tag, model, custom_objects={"tokenizer": tokenizer})
        finally:
            import gc

            # NOTE: We need to free the cache after saving here so that we can load it back later on.
            gc.collect()
            torch.cuda.empty_cache()

    def sanitize_parameters(
        self,
        prompt: str,
        temperature: float | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        repetition_penalty: float | None = None,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        fim_mode = FIM_INDICATOR in prompt
        prefix, suffix = None, None
        if fim_mode:
            try:
                prefix, suffix = prompt.split(FIM_INDICATOR)
            except Exception as err:
                logger.error("Error while processing prompt with FIM mode:\n", exc_info=err)
                raise ValueError(f"Only one {FIM_INDICATOR} allowed in prompt") from err
            prompt_text = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
        else:
            prompt_text = prompt

        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens,
            "repetition_penalty": repetition_penalty,
            # XXX: This value is currently a hack, need more investigate why the
            # default starcoder doesn't include the same value as santacoder EOD
            "pad_token_id": 49152,
            **attrs,
        }

        return prompt_text, generation_config, {}

    def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str:
        return generation_result[0]

    @torch.inference_mode()
    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        inputs = t.cast("torch.Tensor", self.tokenizer.encode(prompt, return_tensors="pt")).to(self.device)
        result_tensor = self.model.generate(
            inputs,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
            # eos_token_id=self.tokenizer.convert_tokens_to_ids("<|end|>"), # NOTE: this is for finetuning starcoder
            generation_config=self.config.model_construct_env(**attrs).to_generation_config(),
        )
        # TODO: We will probably want to return the tokenizer here so that we can manually process this
        # return (skip_special_tokens=False, clean_up_tokenization_spaces=False))
        return self.tokenizer.batch_decode(
            result_tensor[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
