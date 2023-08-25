from __future__ import annotations
import logging, typing as t, openllm
if t.TYPE_CHECKING: import vllm, transformers

logger = logging.getLogger(__name__)
class VLLMDollyV2(openllm.LLM['vllm.LLMEngine', 'transformers.PreTrainedTokenizer']):
  __openllm_internal__ = True
  tokenizer_id = 'local'
