# mypy: disable-error-code="misc"
from __future__ import annotations
import typing as t, transformers
if t.TYPE_CHECKING: import torch, openllm

LogitsProcessorList = transformers.LogitsProcessorList
StoppingCriteriaList = transformers.StoppingCriteriaList
class StopSequenceCriteria(transformers.StoppingCriteria):
  def __init__(self, stop_sequences: str | list[str], tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerBase | transformers.PreTrainedTokenizerFast):
    if isinstance(stop_sequences, str): stop_sequences = [stop_sequences]
    self.stop_sequences, self.tokenizer = stop_sequences, tokenizer
  def __call__(self, input_ids: torch.Tensor, scores: t.Any, **_: t.Any) -> bool: return any(self.tokenizer.decode(input_ids.tolist()[0]).endswith(stop_sequence) for stop_sequence in self.stop_sequences)
class StopOnTokens(transformers.StoppingCriteria):
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_: t.Any) -> bool: return input_ids[0][-1] in {50278, 50279, 50277, 1, 0}
def prepare_logits_processor(config: openllm.LLMConfig) -> transformers.LogitsProcessorList:
  generation_config = config.generation_config
  logits_processor = transformers.LogitsProcessorList()
  if generation_config["temperature"] >= 1e-5 and generation_config["temperature"] != 1.0: logits_processor.append(transformers.TemperatureLogitsWarper(generation_config["temperature"]))
  if generation_config["repetition_penalty"] > 1.0: logits_processor.append(transformers.RepetitionPenaltyLogitsProcessor(generation_config["repetition_penalty"]))
  if 1e-8 <= generation_config["top_p"]: logits_processor.append(transformers.TopPLogitsWarper(generation_config["top_p"]))
  if generation_config["top_k"] > 0: logits_processor.append(transformers.TopKLogitsWarper(generation_config["top_k"]))
  return logits_processor
