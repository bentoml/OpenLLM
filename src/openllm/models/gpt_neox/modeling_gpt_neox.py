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
from .configuration_gpt_neox import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import default_formatter
if t.TYPE_CHECKING: import torch, transformers
else: torch, transformers = openllm.utils.LazyLoader("torch", globals(), "torch"), openllm.utils.LazyLoader("transformers", globals(), "transformers")
logger = logging.getLogger(__name__)
class GPTNeoX(openllm.LLM["transformers.GPTNeoXForCausalLM", "transformers.GPTNeoXTokenizerFast"]):
    __openllm_internal__ = True
    def sanitize_parameters(self, prompt: str, temperature: float | None = None, max_new_tokens: int | None = None, use_default_prompt_template: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        if use_default_prompt_template:
            template_variables = default_formatter.extract_template_variables(DEFAULT_PROMPT_TEMPLATE)
            prompt_variables = {k: v for k, v in attrs.items() if k in template_variables}
            if "instruction" in prompt_variables: raise RuntimeError("'instruction' should be passed as the first argument instead of kwargs when 'use_default_prompt_template=True'")
            try: prompt_text = DEFAULT_PROMPT_TEMPLATE.format(instruction=prompt, **prompt_variables)
            except KeyError as e: raise RuntimeError(f"Missing variable '{e.args[0]}' (required: {template_variables}) in the prompt template. Use 'use_default_prompt_template=False' to disable the default prompt template.") from None
        else: prompt_text = prompt
        return prompt_text, {"max_new_tokens": max_new_tokens, "temperature": temperature}, {}
    @property
    def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]: return {"device_map": "auto" if torch.cuda.device_count() > 1 else None}, {}
    def postprocess_generate(self, prompt: str, generation_result: list[str], **_: t.Any) -> str: return generation_result[0]
    def load_model(self, *args: t.Any, **attrs: t.Any) -> transformers.GPTNeoXForCausalLM:
        model = transformers.AutoModelForCausalLM.from_pretrained(self._bentomodel.path, *args, **attrs)
        if self.config.use_half_precision: model.half()
        return model
    def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
        with torch.inference_mode(): return self.tokenizer.batch_decode(self.model.generate(self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids, do_sample=True, generation_config=self.config.model_construct_env(**attrs).to_generation_config(), pad_token_id=self.tokenizer.eos_token_id, stopping_criteria=transformers.StoppingCriteriaList([openllm.StopOnTokens()])))
