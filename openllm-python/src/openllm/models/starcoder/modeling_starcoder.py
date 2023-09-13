from __future__ import annotations
import typing as t

import bentoml
import openllm
from openllm.utils import generate_labels
from openllm_core.config.configuration_starcoder import EOD
from openllm_core.config.configuration_starcoder import FIM_MIDDLE
from openllm_core.config.configuration_starcoder import FIM_PAD
from openllm_core.config.configuration_starcoder import FIM_PREFIX
from openllm_core.config.configuration_starcoder import FIM_SUFFIX
if t.TYPE_CHECKING: import transformers

class StarCoder(openllm.LLM['transformers.GPTBigCodeForCausalLM', 'transformers.GPT2TokenizerFast']):
  __openllm_internal__ = True

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    import torch
    return {'device_map': 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None, 'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32}, {}

  def import_model(self, *args: t.Any, trust_remote_code: bool = False, **attrs: t.Any) -> bentoml.Model:
    import torch
    import transformers
    torch_dtype, device_map = attrs.pop('torch_dtype', torch.float16), attrs.pop('device_map', 'auto')
    tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_id, **self.llm_parameters[-1])
    tokenizer.add_special_tokens({'additional_special_tokens': [EOD, FIM_PREFIX, FIM_MIDDLE, FIM_SUFFIX, FIM_PAD], 'pad_token': EOD})
    model = transformers.AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch_dtype, device_map=device_map, **attrs)
    try:
      return bentoml.transformers.save_model(self.tag, model, custom_objects={'tokenizer': tokenizer}, labels=generate_labels(self))
    finally:
      torch.cuda.empty_cache()
