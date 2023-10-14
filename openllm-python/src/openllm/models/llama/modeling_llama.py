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
