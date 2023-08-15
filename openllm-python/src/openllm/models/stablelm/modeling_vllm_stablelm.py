from __future__ import annotations
import logging, typing as t, openllm
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
