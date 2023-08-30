from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING:
  import transformers

class ChatGLM(openllm.LLM['transformers.PreTrainedModel', 'transformers.PreTrainedTokenizerFast']):
  __openllm_internal__ = True

  def generate(self, prompt: str, **attrs: t.Any) -> tuple[str, list[tuple[str, str]]]:
    import torch
    with torch.inference_mode():
      self.model.eval()
      # Only use half precision if the model is not yet quantized
      if self.config.use_half_precision: self.model.half()
      return self.model.chat(self.tokenizer, prompt, generation_config=self.config.model_construct_env(**attrs).to_generation_config())

  def embeddings(self, prompts: list[str]) -> openllm.EmbeddingsOutput:
    import torch
    import torch.nn.functional as F
    embeddings: list[list[float]] = []
    num_tokens = 0
    for prompt in prompts:
      input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
      with torch.inference_mode():
        outputs = self.model(input_ids, output_hidden_states=True)
        data = F.normalize(torch.mean(outputs.hidden_states[-1].transpose(0, 1), dim=0), p=2, dim=0)
        embeddings.append(data.tolist())
        num_tokens += len(input_ids[0])
    return openllm.EmbeddingsOutput(embeddings=embeddings, num_tokens=num_tokens)
