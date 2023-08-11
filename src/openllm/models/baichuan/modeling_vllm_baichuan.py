from __future__ import annotations
import logging
import typing as t

import openllm
from openllm._prompt import process_prompt

from .configuration_baichuan import DEFAULT_PROMPT_TEMPLATE

if t.TYPE_CHECKING: import vllm, transformers

logger = logging.getLogger(__name__)
class VLLMBaichuan(openllm.LLM["vllm.LLMEngine", "transformers.PreTrainedTokenizerBase"]):
  __openllm_internal__ = True
  tokenizer_id = "local"
  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, top_p: float | None = None, temperature: float | None = None, use_default_prompt_template: bool = False, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]: return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "top_p": top_p, "temperature": temperature, **attrs}, {}
