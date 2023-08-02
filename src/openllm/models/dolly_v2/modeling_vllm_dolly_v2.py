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
from .configuration_dolly_v2 import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import process_prompt
if t.TYPE_CHECKING:
  import vllm, transformers

logger = logging.getLogger(__name__)

class VLLMDollyV2(openllm.LLM["vllm.LLMEngine", "transformers.PreTrainedTokenizer"]):
  __openllm_internal__ = True
  tokenizer_id = "local"

  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None, top_k: int | None = None, top_p: float | None = None, use_default_prompt_template: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "top_k": top_k, "top_p": top_p, "temperature": temperature, **attrs}, {}
