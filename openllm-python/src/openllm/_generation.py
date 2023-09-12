# mypy: disable-error-code="misc"
from __future__ import annotations
import typing as t

import transformers

if t.TYPE_CHECKING:
  import torch

  import openllm

# reexport from transformers
LogitsProcessorList = transformers.LogitsProcessorList
StoppingCriteriaList = transformers.StoppingCriteriaList

class StopSequenceCriteria(transformers.StoppingCriteria):
  def __init__(self, stop_sequences: str | list[str], tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerBase | transformers.PreTrainedTokenizerFast):
    if isinstance(stop_sequences, str): stop_sequences = [stop_sequences]
    self.stop_sequences, self.tokenizer = stop_sequences, tokenizer

  def __call__(self, input_ids: torch.Tensor, scores: t.Any, **_: t.Any) -> bool:
    return any(self.tokenizer.decode(input_ids.tolist()[0]).endswith(stop_sequence) for stop_sequence in self.stop_sequences)

class StopOnTokens(transformers.StoppingCriteria):
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_: t.Any) -> bool:
    return input_ids[0][-1] in {50278, 50279, 50277, 1, 0}

def prepare_logits_processor(config: openllm.LLMConfig) -> transformers.LogitsProcessorList:
  generation_config = config.generation_config
  logits_processor = transformers.LogitsProcessorList()
  if generation_config['temperature'] >= 1e-5 and generation_config['temperature'] != 1.0:
    logits_processor.append(transformers.TemperatureLogitsWarper(generation_config['temperature']))
  if generation_config['repetition_penalty'] > 1.0:
    logits_processor.append(transformers.RepetitionPenaltyLogitsProcessor(generation_config['repetition_penalty']))
  if 1e-8 <= generation_config['top_p']:
    logits_processor.append(transformers.TopPLogitsWarper(generation_config['top_p']))
  if generation_config['top_k'] > 0: logits_processor.append(transformers.TopKLogitsWarper(generation_config['top_k']))
  return logits_processor

# NOTE: The ordering here is important. Some models have two of these and we have a preference for which value gets used.
SEQLEN_KEYS = ['max_sequence_length', 'seq_length', 'max_position_embeddings', 'max_seq_len', 'model_max_length']

def get_context_length(config: transformers.PretrainedConfig) -> int:
  rope_scaling = getattr(config, 'rope_scaling', None)
  rope_scaling_factor = config.rope_scaling['factor'] if rope_scaling else 1.0
  for key in SEQLEN_KEYS:
    if getattr(config, key, None) is not None: return int(rope_scaling_factor * getattr(config, key))
  return 2048

def is_sentence_complete(output: str) -> bool:
  return output.endswith(('.', '?', '!', '...', '。', '?', '!', '…', '"', "'", '”'))

def is_partial_stop(output: str, stop_str: str) -> bool:
  '''Check whether the output contains a partial stop str.'''
  for i in range(0, min(len(output), len(stop_str))):
    if stop_str.startswith(output[-i:]): return True
  return False
