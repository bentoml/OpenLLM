from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING: import vllm, transformers

class VLLMBaichuan(openllm.LLM['vllm.LLMEngine', 'transformers.PreTrainedTokenizerBase']):
  __openllm_internal__ = True
  tokenizer_id = 'local'
