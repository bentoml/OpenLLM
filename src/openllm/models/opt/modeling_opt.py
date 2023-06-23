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

from ..._prompt import default_formatter
from ...utils import is_peft_available
from .configuration_opt import DEFAULT_PROMPT_TEMPLATE


if t.TYPE_CHECKING:
    import peft
    import torch

    import transformers  # noqa
else:
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")
    peft = openllm.utils.LazyLoader("peft", globals(), "peft")
    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")

logger = logging.getLogger(__name__)


class OPT(openllm.LLM["transformers.OPTForCausalLM", "transformers.GPT2Tokenizer"]):
    __openllm_internal__ = True

    def llm_post_init(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    @property
    def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]] | None:
        model_kwds = {
            "device_map": "auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
        }
        tokenizer_kwds = {
            "padding_side": "left",
            "truncation_side": "left",
        }
        return model_kwds, tokenizer_kwds

    def import_model(
        self,
        model_id: str,
        tag: bentoml.Tag,
        *args: t.Any,
        tokenizer_kwds: dict[str, t.Any],
        **attrs: t.Any,
    ) -> bentoml.Model:
        torch_dtype = attrs.pop("torch_dtype", self.dtype)
        trust_remote_code = attrs.pop("trust_remote_code", False)

        config = transformers.AutoConfig.from_pretrained(model_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, **tokenizer_kwds)
        tokenizer.pad_token_id = config.pad_token_id
        model: transformers.OPTForCausalLM = transformers.AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code, **attrs
        )
        return bentoml.transformers.save_model(tag, model, custom_objects={"tokenizer": tokenizer})

    def load_model(self, tag: bentoml.Tag, *args: t.Any, **attrs: t.Any) -> transformers.OPTForCausalLM:
        torch_dtype = attrs.pop("torch_dtype", self.dtype)
        trust_remote_code = attrs.pop("trust_remote_code", False)

        _ref = bentoml.transformers.get(tag)
        model: transformers.OPTForCausalLM = transformers.AutoModelForCausalLM.from_pretrained(
            _ref.path, trust_remote_code=trust_remote_code, torch_dtype=torch_dtype, **attrs
        )
        return model

    def sanitize_parameters(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        num_return_sequences: int | None = None,
        use_default_prompt_template: bool = False,
        **attrs: t.Any,
    ) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        if use_default_prompt_template:
            template_variables = default_formatter.extract_template_variables(DEFAULT_PROMPT_TEMPLATE)
            prompt_variables = {k: v for k, v in attrs.items() if k in template_variables}
            if "instruction" in prompt_variables:
                raise RuntimeError(
                    "'instruction' should be passed as the first argument "
                    "instead of kwargs when 'use_default_prompt_template=True'"
                )
            try:
                prompt_text = DEFAULT_PROMPT_TEMPLATE.format(instruction=prompt, **prompt_variables)
            except KeyError as e:
                raise RuntimeError(
                    f"Missing variable '{e.args[0]}' (required: {template_variables}) in the prompt template. "
                    "Use 'use_default_prompt_template=False' to disable the default prompt template."
                )
        else:
            prompt_text = prompt

        generation_config = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "num_return_sequences": num_return_sequences,
        }
        return prompt_text, generation_config, {}

    def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **attrs: t.Any) -> str:
        if len(generation_result) == 1:
            if self.config.format_outputs:
                logger.warning("'format_outputs' doesn't have any effect when 'num_return_sequences=1'")
            return generation_result[0]

        if self.config.format_outputs:
            return "Generated result:\n" + "\n -".join(generation_result)
        else:
            return "\n".join(generation_result)

    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        with torch.inference_mode():
            if torch.cuda.is_available() and torch.cuda.device_count() == 1:
                self.model.cuda()

            if is_peft_available() and isinstance(self.model, peft.PeftModel):
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            else:
                inputs = {
                    "inputs": t.cast(torch.Tensor, self.tokenizer(prompt, return_tensors="pt").input_ids).to(
                        self.device
                    )
                }

            generated_tensors = self.model.generate(
                **inputs,
                do_sample=True,
                generation_config=self.config.model_construct_env(**attrs).to_generation_config(),
            )
            return self.tokenizer.batch_decode(generated_tensors, skip_special_tokens=True)
