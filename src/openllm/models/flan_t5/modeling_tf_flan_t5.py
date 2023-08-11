from __future__ import annotations
import sys, typing as t

import openllm
from openllm._prompt import process_prompt

from .configuration_flan_t5 import DEFAULT_PROMPT_TEMPLATE

if t.TYPE_CHECKING: import transformers

class TFFlanT5(openllm.LLM["transformers.TFT5ForConditionalGeneration", "transformers.T5TokenizerFast"]):
  __openllm_internal__ = True
  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None, top_k: int | None = None, top_p: float | None = None, repetition_penalty: float | None = None, use_default_prompt_template: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]: return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_k": top_k, "top_p": top_p, "repetition_penalty": repetition_penalty}, {}
  def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str: return generation_result[0]
  def generate(self, prompt: str, **attrs: t.Any) -> list[str]: return self.tokenizer.batch_decode(self.model.generate(self.tokenizer(prompt, return_tensors="tf").input_ids, do_sample=True, generation_config=self.config.model_construct_env(**attrs).to_generation_config()), skip_special_tokens=True)
