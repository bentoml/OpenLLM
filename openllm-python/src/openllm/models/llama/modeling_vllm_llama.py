from __future__ import annotations
import typing as t

import openllm
if t.TYPE_CHECKING: import vllm, transformers

class VLLMLlama(openllm.LLM['vllm.LLMEngine', 'transformers.LlamaTokenizerFast']):
  __openllm_internal__ = True
