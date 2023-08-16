from __future__ import annotations
import logging, typing as t, openllm
from openllm._prompt import process_prompt
from .configuration_falcon import DEFAULT_PROMPT_TEMPLATE
if t.TYPE_CHECKING: import vllm, transformers

logger = logging.getLogger(__name__)
class VLLMFalcon(openllm.LLM["vllm.LLMEngine", "transformers.PreTrainedTokenizerBase"]):
  __openllm_internal__ = True
  tokenizer_id = "local"
  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, top_k: int | None = None, num_return_sequences: int | None = None, eos_token_id: int | None = None, use_default_prompt_template: bool = False, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]: return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "top_k": top_k, "num_return_sequences": num_return_sequences, "eos_token_id": eos_token_id, **attrs}, {}
