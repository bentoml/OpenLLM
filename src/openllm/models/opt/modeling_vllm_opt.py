
from __future__ import annotations
import logging
import typing as t
import openllm
from .configuration_opt import DEFAULT_PROMPT_TEMPLATE
from ..._prompt import process_prompt
if t.TYPE_CHECKING:
  import vllm, transformers

logger = logging.getLogger(__name__)

class VLLMOPT(openllm.LLM["vllm.LLMEngine", "transformers.GPT2Tokenizer"]):
  __openllm_internal__ = True
  tokenizer_id = "local"

  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None, top_k: int | None = None, num_return_sequences: int | None = None, use_default_prompt_template: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_k": top_k, "num_return_sequences": num_return_sequences}, {}
