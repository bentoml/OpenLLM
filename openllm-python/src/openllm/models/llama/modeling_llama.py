from __future__ import annotations
import typing as t, openllm
if t.TYPE_CHECKING: import transformers

class Llama(openllm.LLM["transformers.LlamaForCausalLM", "transformers.LlamaTokenizerFast"]):
  __openllm_internal__ = True
  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    import torch
    return {"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32}, {}
  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    import torch
    with torch.inference_mode(): return self.tokenizer.batch_decode(self.model.generate(**self.tokenizer(prompt, return_tensors="pt").to(self.device), generation_config=self.config.model_construct_env(**attrs).to_generation_config(), do_sample=True, stopping_criteria=openllm.StoppingCriteriaList([openllm.StopOnTokens()])), skip_special_tokens=True, clean_up_tokenization_spaces=True)
  def embeddings(self, prompts: list[str]) -> openllm.LLMEmbeddings:
    import torch, torch.nn.functional as F
    encoding = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    with torch.inference_mode():
      data = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
      mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
      masked_embeddings = data * mask
      sum_embeddings, seq_length = torch.sum(masked_embeddings, dim=1), torch.sum(mask, dim=1)
    return openllm.LLMEmbeddings(embeddings=F.normalize(sum_embeddings / seq_length, p=2, dim=1).tolist(), num_tokens=int(torch.sum(attention_mask).item()))
