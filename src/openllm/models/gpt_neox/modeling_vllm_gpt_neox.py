
from __future__ import annotations
import logging
import typing as t
import openllm
from .configuration_gpt_neox import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import process_prompt
if t.TYPE_CHECKING: import vllm, transformers

logger = logging.getLogger(__name__)

class VLLMGPTNeoX(openllm.LLM["vllm.LLMEngine", "transformers.GPTNeoXTokenizerFast"]):
  __openllm_internal__ = True
  tokenizer_id = "local"

  def sanitize_parameters(self, prompt: str, temperature: float | None = None, max_new_tokens: int | None = None, use_default_prompt_template: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "temperature": temperature}, {}
