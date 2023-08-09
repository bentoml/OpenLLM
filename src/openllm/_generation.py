# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generation utilities to be reused throughout."""
from __future__ import annotations
import typing as t

import transformers

if t.TYPE_CHECKING:
  from abc import ABC
  from abc import abstractmethod

  import torch

  import openllm

  class LogitsProcessorList(list[transformers.LogitsProcessor | transformers.LogitsWarper]):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: t.Any) -> torch.FloatTensor:
      ...

  class StoppingCriteriaList(list[transformers.StoppingCriteria]):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: t.Any) -> bool:
      ...

  class StoppingCriteria(ABC):
    @abstractmethod
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: t.Any) -> bool:
      ...
else:
  LogitsProcessorList = transformers.LogitsProcessorList
  StoppingCriteriaList = transformers.StoppingCriteriaList
  StoppingCriteria = transformers.StoppingCriteria

class StopSequenceCriteria(StoppingCriteria):
  def __init__(self, stop_sequences: str | list[str], tokenizer: transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerBase | transformers.PreTrainedTokenizerFast):
    if isinstance(stop_sequences, str): stop_sequences = [stop_sequences]
    self.stop_sequences, self.tokenizer = stop_sequences, tokenizer
  def __call__(self, input_ids: torch.Tensor, scores: t.Any, **_: t.Any) -> bool: return any(self.tokenizer.decode(input_ids.tolist()[0]).endswith(stop_sequence) for stop_sequence in self.stop_sequences)
class StopOnTokens(StoppingCriteria):
  def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **_: t.Any) -> bool: return t.cast(int, input_ids[0][-1]) in {50278, 50279, 50277, 1, 0}

def prepare_logits_processor(config: openllm.LLMConfig) -> LogitsProcessorList:
  generation_config = config.generation_config
  logits_processor = LogitsProcessorList()
  if generation_config["temperature"] >= 1e-5 and generation_config["temperature"] != 1.0: logits_processor.append(transformers.TemperatureLogitsWarper(generation_config["temperature"]))
  if generation_config["repetition_penalty"] > 1.0: logits_processor.append(transformers.RepetitionPenaltyLogitsProcessor(generation_config["repetition_penalty"]))
  if 1e-8 <= generation_config["top_p"]: logits_processor.append(transformers.TopPLogitsWarper(generation_config["top_p"]))
  if generation_config["top_k"] > 0: logits_processor.append(transformers.TopKLogitsWarper(generation_config["top_k"]))
  return logits_processor
