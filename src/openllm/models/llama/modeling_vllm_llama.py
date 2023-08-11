from __future__ import annotations
import logging
import typing as t

import openllm
from openllm._prompt import process_prompt

from .configuration_llama import DEFAULT_PROMPT_TEMPLATE

if t.TYPE_CHECKING: import vllm, transformers
logger = logging.getLogger(__name__)

class VLLMLlama(openllm.LLM["vllm.LLMEngine", "transformers.LlamaTokenizerFast"]):
  __openllm_internal__ = True
  def sanitize_parameters(self, prompt: str, top_k: int | None = None, top_p: float | None = None, temperature: float | None = None, max_new_tokens: int | None = None, use_default_prompt_template: bool = False, use_llama2_prompt: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]: return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE("v2" if use_llama2_prompt else "v1") if use_default_prompt_template else None, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k}, {}
