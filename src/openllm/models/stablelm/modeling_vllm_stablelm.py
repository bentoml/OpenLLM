# Copyright 2023 BentoML Team. All rights reserved.
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
from openllm._prompt import process_prompt

from .configuration_stablelm import DEFAULT_PROMPT_TEMPLATE, SYSTEM_PROMPT

if t.TYPE_CHECKING: import vllm, transformers

logger = logging.getLogger(__name__)
class VLLMStableLM(openllm.LLM["vllm.LLMEngine", "transformers.GPTNeoXTokenizerFast"]):
  __openllm_internal__ = True
  tokenizer_id = "local"
  def sanitize_parameters(self, prompt: str, temperature: float | None = None, max_new_tokens: int | None = None, top_k: int | None = None, top_p: float | None = None, use_default_prompt_template: bool = False, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    if "tuned" in self._model_id and use_default_prompt_template:
      system_prompt = attrs.pop("system_prompt", SYSTEM_PROMPT)
      prompt_text = process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, system_prompt=system_prompt, **attrs)
    else: prompt_text = prompt
    return prompt_text, {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_k": top_k, "top_p": top_p}, {}
