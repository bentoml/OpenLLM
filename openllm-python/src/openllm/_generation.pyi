from typing import Any, List, Union

from torch import FloatTensor, LongTensor
from transformers import (
  LogitsProcessorList,
  PretrainedConfig,
  PreTrainedTokenizer,
  PreTrainedTokenizerBase,
  PreTrainedTokenizerFast,
)

from openllm_core import LLMConfig

Tokenizer = Union[PreTrainedTokenizerBase, PreTrainedTokenizer, PreTrainedTokenizerFast]

class StopSequenceCriteria:
  stop_sequences: List[str]
  tokenizer: Tokenizer
  def __init__(self, stop_sequences: Union[str, List[str]], tokenizer: Tokenizer) -> None: ...
  def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs: Any) -> bool: ...

class StopOnTokens:
  def __call__(self, input_ids: LongTensor, scores: FloatTensor, **kwargs: Any) -> bool: ...

def prepare_logits_processor(config: LLMConfig) -> LogitsProcessorList: ...
def get_context_length(config: PretrainedConfig) -> int: ...
def is_sentence_complete(output: str) -> bool: ...
def is_partial_stop(output: str, stop_str: str) -> bool: ...
