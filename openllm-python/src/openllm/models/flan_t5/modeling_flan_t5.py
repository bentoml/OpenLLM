from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING:
  import transformers

class FlanT5(openllm.LLM['transformers.T5ForConditionalGeneration', 'transformers.T5TokenizerFast']):
  __openllm_internal__ = True

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    import torch
    with torch.inference_mode():
      return self.tokenizer.batch_decode(self.model.generate(**self.tokenizer(prompt, return_tensors='pt').to(self.device),
                                                             do_sample=True,
                                                             generation_config=self.config.model_construct_env(**attrs).to_generation_config()),
                                         skip_special_tokens=True)

  def embeddings(self, prompts: list[str]) -> openllm.EmbeddingsOutput:
    import torch
    import torch.nn.functional as F
    embeddings: list[list[float]] = []
    num_tokens = 0
    for prompt in prompts:
      input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
      with torch.inference_mode():
        outputs = self.model(input_ids, decoder_input_ids=input_ids)
        data = F.normalize(torch.mean(outputs.encoder_last_hidden_state[0], dim=0), p=2, dim=0)
        embeddings.append(data.tolist())
        num_tokens += len(input_ids[0])
    return openllm.EmbeddingsOutput(embeddings=embeddings, num_tokens=num_tokens)
