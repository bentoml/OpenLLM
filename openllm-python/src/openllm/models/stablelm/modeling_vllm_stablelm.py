from __future__ import annotations
import logging
import typing as t

import openllm
if t.TYPE_CHECKING: import vllm, transformers

class VLLMStableLM(openllm.LLM['vllm.LLMEngine', 'transformers.GPTNeoXTokenizerFast']):
  __openllm_internal__ = True
  tokenizer_id = 'local'
