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
