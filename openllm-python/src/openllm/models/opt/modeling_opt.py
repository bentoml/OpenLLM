from __future__ import annotations
import logging
import typing as t

import openllm
if t.TYPE_CHECKING: import transformers

logger = logging.getLogger(__name__)

class OPT(openllm.LLM['transformers.OPTForCausalLM', 'transformers.GPT2Tokenizer']):
  __openllm_internal__ = True

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    import torch
    return {'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32}, {}

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    import torch
    with torch.inference_mode():
      return self.tokenizer.batch_decode(self.model.generate(**self.tokenizer(prompt, return_tensors='pt').to(self.device),
                                                             do_sample=True,
                                                             generation_config=self.config.model_construct_env(**attrs).to_generation_config()),
                                         skip_special_tokens=True)
