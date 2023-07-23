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
from ...utils import generate_labels
if t.TYPE_CHECKING:
    import torch, transformers
else:
    torch, transformers = openllm.utils.LazyLoader("torch", globals(), "torch"), openllm.utils.LazyLoader("transformers", globals(), "transformers")
logger = logging.getLogger(__name__)
FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD, EOD, FIM_INDICATOR = "<fim-prefix>", "<fim-middle>", "<fim-suffix>", "<fim-pad>", "<|endoftext|>", "<FILL_HERE>"
class StarCoder(openllm.LLM["transformers.GPTBigCodeForCausalLM", "transformers.GPT2TokenizerFast"]):
    __openllm_internal__ = True
    @property
    def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]: return {"device_map": "auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None, "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}, {"padding_side": "left"}
    def import_model(self, *args: t.Any, trust_remote_code: bool = False, **attrs: t.Any) -> bentoml.Model:
        torch_dtype, device_map = attrs.pop("torch_dtype", torch.float16), attrs.pop("device_map", "auto")
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, **self.llm_parameters[-1])
        tokenizer.add_special_tokens({"additional_special_tokens": [EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD], "pad_token": EOD})
        model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch_dtype, device_map=device_map, **attrs)
        try: return bentoml.transformers.save_model(self.tag, model, custom_objects={"tokenizer": tokenizer}, labels=generate_labels(self))
        finally: torch.cuda.empty_cache()
    def sanitize_parameters(self, prompt: str, temperature: float | None = None, top_p: float | None = None, max_new_tokens: int | None = None, repetition_penalty: float | None = None, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        fim_mode, prefix, suffix = FIM_INDICATOR in prompt, None, None
        if fim_mode:
            try: prefix, suffix = prompt.split(FIM_INDICATOR)
            except Exception as err: raise ValueError(f"Only one {FIM_INDICATOR} allowed in prompt") from err
            prompt_text = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
        else: prompt_text = prompt
        # XXX: This value for pad_token_id is currently a hack, need more investigate why the
        # default starcoder doesn't include the same value as santacoder EOD
        return prompt_text, {"temperature": temperature, "top_p": top_p, "max_new_tokens": max_new_tokens, "repetition_penalty": repetition_penalty, "pad_token_id": 49152, **attrs}, {}

    def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str: return generation_result[0]
    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        with torch.inference_mode():
            # eos_token_id=self.tokenizer.convert_tokens_to_ids("<|end|>"), # NOTE: this is for finetuning starcoder
            # NOTE: support fine-tuning starcoder
            result_tensor = self.model.generate(self.tokenizer.encode(prompt, return_tensors="pt").to(self.device), do_sample=True, pad_token_id=self.tokenizer.eos_token_id, generation_config=self.config.model_construct_env(**attrs).to_generation_config())
            # TODO: We will probably want to return the tokenizer here so that we can manually process this
            # return (skip_special_tokens=False, clean_up_tokenization_spaces=False))
            return self.tokenizer.batch_decode(result_tensor[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    def generate_one(self, prompt: str, stop: list[str], **preprocess_generate_kwds: t.Any) -> list[dict[t.Literal["generated_text"], str]]:
        max_new_tokens, encoded_inputs = preprocess_generate_kwds.pop("max_new_tokens", 200), self.tokenizer(prompt, return_tensors="pt").to(self.device)
        src_len, stopping_criteria = encoded_inputs["input_ids"].shape[1], preprocess_generate_kwds.pop("stopping_criteria", transformers.StoppingCriteriaList([]))
        stopping_criteria.append(openllm.StopSequenceCriteria(stop, self.tokenizer))
        result = self.tokenizer.decode(self.model.generate(encoded_inputs["input_ids"], max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)[0].tolist()[src_len:])
        # Inference API returns the stop sequence
        for stop_seq in stop:
            if result.endswith(stop_seq): result = result[: -len(stop_seq)]
        return [{"generated_text": result}]
