from __future__ import annotations
import logging, typing as t, openllm
from openllm._prompt import process_prompt
from .configuration_dolly_v2 import DEFAULT_PROMPT_TEMPLATE
if t.TYPE_CHECKING: import vllm, transformers

logger = logging.getLogger(__name__)
class VLLMDollyV2(openllm.LLM["vllm.LLMEngine", "transformers.PreTrainedTokenizer"]):
  __openllm_internal__ = True
  tokenizer_id = "local"
  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None, top_k: int | None = None, top_p: float | None = None, use_default_prompt_template: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]: return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "top_k": top_k, "top_p": top_p, "temperature": temperature, **attrs}, {}
