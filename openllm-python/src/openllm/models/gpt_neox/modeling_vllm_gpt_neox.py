from __future__ import annotations
import typing as t, openllm
if t.TYPE_CHECKING: import vllm, transformers
class VLLMGPTNeoX(openllm.LLM['vllm.LLMEngine', 'transformers.GPTNeoXTokenizerFast']):
  __openllm_internal__ = True
  tokenizer_id = 'local'
