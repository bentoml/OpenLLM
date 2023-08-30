from __future__ import annotations
import logging
import typing as t

import openllm
if t.TYPE_CHECKING: import transformers

logger = logging.getLogger(__name__)

class GPTNeoX(openllm.LLM['transformers.GPTNeoXForCausalLM', 'transformers.GPTNeoXTokenizerFast']):
  __openllm_internal__ = True

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    import torch
    return {'device_map': 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None}, {}

  def load_model(self, *args: t.Any, **attrs: t.Any) -> transformers.GPTNeoXForCausalLM:
    import transformers
    model = transformers.AutoModelForCausalLM.from_pretrained(self._bentomodel.path, *args, **attrs)
    if self.config.use_half_precision: model.half()
    return model

  def generate(self, prompt: str, **attrs: t.Any) -> list[str]:
    import torch
    with torch.inference_mode():
      return self.tokenizer.batch_decode(
          self.model.generate(self.tokenizer(prompt, return_tensors='pt').to(self.device).input_ids,
                              do_sample=True,
                              generation_config=self.config.model_construct_env(**attrs).to_generation_config(),
                              pad_token_id=self.tokenizer.eos_token_id,
                              stopping_criteria=openllm.StoppingCriteriaList([openllm.StopOnTokens()])))
