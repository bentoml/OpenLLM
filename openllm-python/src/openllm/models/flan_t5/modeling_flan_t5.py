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
