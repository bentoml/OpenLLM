from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING: import transformers

class Baichuan(openllm.LLM['transformers.PreTrainedModel', 'transformers.PreTrainedTokenizerBase']):
  __openllm_internal__ = True

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    import torch
    inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):  # type: ignore[attr-defined]
      outputs = self.model.generate(**inputs, generation_config=self.config.model_construct_env(**attrs).to_generation_config())
      return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
