from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING:
  import transformers

class Llama(openllm.LLM['transformers.LlamaForCausalLM', 'transformers.LlamaTokenizerFast']):
  __openllm_internal__ = True

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    import torch
    return {'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32}, {}

  def embeddings(self, prompts: list[str]) -> openllm.EmbeddingsOutput:
    import torch
    import torch.nn.functional as F
    encoding = self.tokenizer(prompts, padding=True, return_tensors='pt').to(self.device)
    input_ids, attention_mask = encoding['input_ids'], encoding['attention_mask']
    with torch.inference_mode():
      data = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
      mask = attention_mask.unsqueeze(-1).expand(data.size()).float()
      masked_embeddings = data * mask
      sum_embeddings, seq_length = torch.sum(masked_embeddings, dim=1), torch.sum(mask, dim=1)
    return openllm.EmbeddingsOutput(embeddings=F.normalize(sum_embeddings / seq_length, p=2, dim=1).tolist(), num_tokens=int(torch.sum(attention_mask).item()))

  def generate_one(self, prompt: str, stop: list[str], **preprocess_generate_kwds: t.Any) -> list[dict[t.Literal['generated_text'], str]]:
    max_new_tokens, encoded_inputs = preprocess_generate_kwds.pop('max_new_tokens', 200), self.tokenizer(prompt, return_tensors='pt').to(self.device)
    src_len, stopping_criteria = encoded_inputs['input_ids'].shape[1], preprocess_generate_kwds.pop('stopping_criteria', openllm.StoppingCriteriaList([]))
    stopping_criteria.append(openllm.StopSequenceCriteria(stop, self.tokenizer))
    result = self.tokenizer.decode(self.model.generate(encoded_inputs['input_ids'], max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)[0].tolist()[src_len:])
    # Inference API returns the stop sequence
    for stop_seq in stop:
      if result.endswith(stop_seq): result = result[:-len(stop_seq)]
    return [{'generated_text': result}]
