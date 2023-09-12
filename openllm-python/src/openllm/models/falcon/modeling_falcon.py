from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING: import torch, transformers
else:
  torch, transformers = openllm.utils.LazyLoader('torch', globals(), 'torch'), openllm.utils.LazyLoader('transformers', globals(), 'transformers')

class Falcon(openllm.LLM['transformers.PreTrainedModel', 'transformers.PreTrainedTokenizerBase']):
  __openllm_internal__ = True

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    return {'torch_dtype': torch.bfloat16, 'device_map': 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None}, {}

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    eos_token_id, inputs = attrs.pop('eos_token_id', self.tokenizer.eos_token_id), self.tokenizer(prompt, return_tensors='pt').to(self.device)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):  # type: ignore[attr-defined]
      return self.tokenizer.batch_decode(self.model.generate(input_ids=inputs['input_ids'],
                                                             attention_mask=inputs['attention_mask'],
                                                             generation_config=self.config.model_construct_env(eos_token_id=eos_token_id, **attrs).to_generation_config()),
                                         skip_special_tokens=True)
