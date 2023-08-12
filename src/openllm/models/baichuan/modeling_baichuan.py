from __future__ import annotations
import typing as t, openllm
from openllm._prompt import process_prompt
from .configuration_baichuan import DEFAULT_PROMPT_TEMPLATE

if t.TYPE_CHECKING: import torch, transformers
else: torch, transformers = openllm.utils.LazyLoader("torch", globals(), "torch"), openllm.utils.LazyLoader("transformers", globals(), "transformers")

class Baichuan(openllm.LLM["transformers.PreTrainedModel", "transformers.PreTrainedTokenizerBase"]):
  __openllm_internal__ = True
  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, top_p: float | None = None, temperature: float | None = None, use_default_prompt_template: bool = False, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]: return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "top_p": top_p, "temperature": temperature, **attrs}, {}
  def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str: return generation_result[0]
  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
      outputs = self.model.generate(**inputs, generation_config=self.config.model_construct_env(**attrs).to_generation_config())
      return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
