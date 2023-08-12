from __future__ import annotations
import typing as t, openllm
from openllm._prompt import process_prompt
from .configuration_flan_t5 import DEFAULT_PROMPT_TEMPLATE
if t.TYPE_CHECKING: import torch, transformers, torch.nn.functional as F
else: torch, transformers, F = openllm.utils.LazyLoader("torch", globals(), "torch"), openllm.utils.LazyLoader("transformers", globals(), "transformers"), openllm.utils.LazyLoader("F", globals(), "torch.nn.functional")

class FlanT5(openllm.LLM["transformers.T5ForConditionalGeneration", "transformers.T5TokenizerFast"]):
  __openllm_internal__ = True
  def sanitize_parameters(self, prompt: str, max_new_tokens: int | None = None, temperature: float | None = None, top_k: int | None = None, top_p: float | None = None, repetition_penalty: float | None = None, use_default_prompt_template: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]: return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_k": top_k, "top_p": top_p, "repetition_penalty": repetition_penalty}, {}
  def postprocess_generate(self, prompt: str, generation_result: t.Sequence[str], **_: t.Any) -> str: return generation_result[0]
  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    with torch.inference_mode(): return self.tokenizer.batch_decode(self.model.generate(**self.tokenizer(prompt, return_tensors="pt").to(self.device), do_sample=True, generation_config=self.config.model_construct_env(**attrs).to_generation_config()), skip_special_tokens=True)
  def embeddings(self, prompts: list[str]) -> openllm.LLMEmbeddings:
    embeddings: list[list[float]] = []
    num_tokens = 0
    for prompt in prompts:
      input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
      with torch.inference_mode():
        outputs = self.model(input_ids, decoder_input_ids=input_ids)
        data = F.normalize(torch.mean(outputs.encoder_last_hidden_state[0], dim=0), p=2, dim=0)
        embeddings.append(data.tolist())
        num_tokens += len(input_ids[0])
    return openllm.LLMEmbeddings(embeddings=embeddings, num_tokens=num_tokens)
