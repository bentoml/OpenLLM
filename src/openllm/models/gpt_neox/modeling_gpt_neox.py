from __future__ import annotations
import logging
import typing as t

import openllm
from openllm._prompt import process_prompt

from .configuration_gpt_neox import DEFAULT_PROMPT_TEMPLATE

if t.TYPE_CHECKING: import torch, transformers
else: torch, transformers = openllm.utils.LazyLoader("torch", globals(), "torch"), openllm.utils.LazyLoader("transformers", globals(), "transformers")
logger = logging.getLogger(__name__)

class GPTNeoX(openllm.LLM["transformers.GPTNeoXForCausalLM", "transformers.GPTNeoXTokenizerFast"]):
  __openllm_internal__ = True
  def sanitize_parameters(self, prompt: str, temperature: float | None = None, max_new_tokens: int | None = None, use_default_prompt_template: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]: return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "temperature": temperature}, {}
  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]: return {"device_map": "auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None}, {}
  def postprocess_generate(self, prompt: str, generation_result: list[str], **_: t.Any) -> str: return generation_result[0]
  def load_model(self, *args: t.Any, **attrs: t.Any) -> transformers.GPTNeoXForCausalLM:
    model = transformers.AutoModelForCausalLM.from_pretrained(self._bentomodel.path, *args, **attrs)
    if self.config.use_half_precision: model.half()
    return model
  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    with torch.inference_mode(): return self.tokenizer.batch_decode(self.model.generate(self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids, do_sample=True, generation_config=self.config.model_construct_env(**attrs).to_generation_config(), pad_token_id=self.tokenizer.eos_token_id, stopping_criteria=openllm.StoppingCriteriaList([openllm.StopOnTokens()])))
