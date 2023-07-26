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
from .configuration_llama import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import default_formatter
if t.TYPE_CHECKING: import vllm, transformers
logger = logging.getLogger(__name__)
class VLLMLlama(openllm.LLM["vllm.LLMEngine", "transformers.LlamaTokenizerFast"]):
    __openllm_internal__ = True
    def sanitize_parameters(self, prompt: str, top_k: int | None = None, top_p: float | None = None, temperature: float | None = None, max_new_tokens: int | None = None, use_default_prompt_template: bool = False, use_llama2_prompt: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
        if use_default_prompt_template:
            _PROMPT = DEFAULT_PROMPT_TEMPLATE("v2" if use_llama2_prompt else "v1")
            template_variables = default_formatter.extract_template_variables(_PROMPT)
            prompt_variables = {k: v for k, v in attrs.items() if k in template_variables}
            if "instruction" in prompt_variables: raise RuntimeError("'instruction' should be passed as the first argument instead of kwargs when 'use_default_prompt_template=True'")
            try: prompt_text = _PROMPT.format(instruction=prompt, **prompt_variables)
            except KeyError as e: raise RuntimeError(f"Missing variable '{e.args[0]}' (required: {template_variables}) in the prompt template. Use 'use_default_prompt_template=False' to disable the default prompt template.") from None
        else: prompt_text = prompt
        return prompt_text, {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k}, {}
