from __future__ import annotations
import logging, typing as t, openllm
from openllm._prompt import process_prompt
from .configuration_llama import DEFAULT_PROMPT_TEMPLATE
if t.TYPE_CHECKING: import torch, transformers, torch.nn.functional as F
else: torch, transformers, F = openllm.utils.LazyLoader("torch", globals(), "torch"), openllm.utils.LazyLoader("transformers", globals(), "transformers"), openllm.utils.LazyLoader("F", globals(), "torch.nn.functional")

logger = logging.getLogger(__name__)
class Llama(openllm.LLM["transformers.LlamaForCausalLM", "transformers.LlamaTokenizerFast"]):
  __openllm_internal__ = True
  def sanitize_parameters(self, prompt: str, top_k: int | None = None, top_p: float | None = None, temperature: float | None = None, max_new_tokens: int | None = None, use_default_prompt_template: bool = False, use_llama2_prompt: bool = True, **attrs: t.Any) -> tuple[str, dict[str, t.Any], dict[str, t.Any]]: return process_prompt(prompt, DEFAULT_PROMPT_TEMPLATE("v2" if use_llama2_prompt else "v1") if use_default_prompt_template else None, use_default_prompt_template, **attrs), {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": top_p, "top_k": top_k}, {}
  def postprocess_generate(self, prompt: str, generation_result: list[str], **_: t.Any) -> str: return generation_result[0]
  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    with torch.inference_mode(): return self.tokenizer.batch_decode(self.model.generate(**self.tokenizer(prompt, return_tensors="pt").to(self.device), generation_config=self.config.model_construct_env(**attrs).to_generation_config(), do_sample=True, stopping_criteria=openllm.StoppingCriteriaList([openllm.StopOnTokens()])), skip_special_tokens=True, clean_up_tokenization_spaces=True)
  def embeddings(self, prompts: list[str]) -> openllm.LLMEmbeddings:
    encoding = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    with torch.inference_mode():
      data = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
      mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
      masked_embeddings = data * mask
      sum_embeddings, seq_length = torch.sum(masked_embeddings, dim=1), torch.sum(mask, dim=1)
    return openllm.LLMEmbeddings(embeddings=F.normalize(sum_embeddings / seq_length, p=2, dim=1).tolist(), num_tokens=torch.sum(attention_mask).item())
