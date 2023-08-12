from __future__ import annotations
import logging, typing as t, openllm
from openllm._prompt import process_prompt
from .configuration_mpt import DEFAULT_PROMPT_TEMPLATE, MPTPromptType
if t.TYPE_CHECKING: import transformers, vllm

logger = logging.getLogger(__name__)
class VLLMMPT(openllm.LLM["vllm.LLMEngine", "transformers.GPTNeoXTokenizerFast"]):
  __openllm_internal__ = True
  tokenizer_id = "local"
  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None, top_p: float | None = None, prompt_type: MPTPromptType | None = None, use_default_prompt_template: bool = True, **attrs: t.Any,) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]:
    _template = None
    if use_default_prompt_template:
      if prompt_type is None:
        if "instruct" in self.model_id: prompt_type = "instruct"
        elif "storywriter" in self.model_id: prompt_type = "storywriter"
        elif "chat" in self.model_id: prompt_type = "chat"
        else: prompt_type = "default"
      _template = DEFAULT_PROMPT_TEMPLATE(prompt_type)
    return process_prompt(prompt, _template, use_default_prompt_template), {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p}, {}
