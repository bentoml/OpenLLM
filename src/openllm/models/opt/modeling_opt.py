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
import openllm
from .configuration_opt import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import default_formatter
if t.TYPE_CHECKING:
    import torch, transformers
else:
    torch, transformers = openllm.utils.LazyLoader("torch", globals(), "torch"), openllm.utils.LazyLoader("transformers", globals(), "transformers")
logger = logging.getLogger(__name__)
class OPT(openllm.LLM["transformers.OPTForCausalLM", "transformers.GPT2Tokenizer"]):
    __openllm_internal__ = True
    def llm_post_init(self): self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    @property
    def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]: return {"device_map": "auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None, "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}, {"padding_side": "left", "truncation_side": "left"}
    def load_model(self, *args: t.Any, **attrs: t.Any) -> transformers.OPTForCausalLM:
        torch_dtype = attrs.pop("torch_dtype", self.dtype)
        model = transformers.AutoModelForCausalLM.from_pretrained(self._bentomodel.path, *args, torch_dtype=torch_dtype, **attrs)
        return model
    def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None, top_k: int | None = None, num_return_sequences: int | None = None, use_default_prompt_template: bool = False, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        if use_default_prompt_template:
            template_variables = default_formatter.extract_template_variables(DEFAULT_PROMPT_TEMPLATE)
            prompt_variables = {k: v for k, v in attrs.items() if k in template_variables}
            if "instruction" in prompt_variables: raise RuntimeError("'instruction' should be passed as the first argument instead of kwargs when 'use_default_prompt_template=True'")
            try: prompt_text = DEFAULT_PROMPT_TEMPLATE.format(instruction=prompt, **prompt_variables)
            except KeyError as e: raise RuntimeError(f"Missing variable '{e.args[0]}' (required: {template_variables}) in the prompt template. Use 'use_default_prompt_template=False' to disable the default prompt template.") from None
        else: prompt_text = prompt
        return prompt_text, {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_k": top_k, "num_return_sequences": num_return_sequences}, {}
    def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **attrs: t.Any) -> str:
        if len(generation_result) == 1: return generation_result[0]
        if self.config.format_outputs: return "Generated result:\n" + "\n -".join(generation_result)
        else: return "\n".join(generation_result)
    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        with torch.inference_mode(): return self.tokenizer.batch_decode(self.model.generate(**self.tokenizer(prompt, return_tensors="pt").to(self.device), do_sample=True, generation_config=self.config.model_construct_env(**attrs).to_generation_config()), skip_special_tokens=True)
