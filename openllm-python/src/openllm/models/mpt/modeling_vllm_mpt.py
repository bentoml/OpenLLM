from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING: import transformers, vllm

class VLLMMPT(openllm.LLM['vllm.LLMEngine', 'transformers.GPTNeoXTokenizerFast']):
  __openllm_internal__ = True
  tokenizer_id = 'local'
