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
    import torch


class StopSequenceCriteria(transformers.StoppingCriteria):
    """This class used to stop generation when a seq of tokens are met.

    Args:
        stop_sequences: `str` or `list[str]` of the sequence (list of sequences) on which to stop execution.
        tokenizer: Tokenizer to be used to decode the model outputs.
    """

    def __init__(self, stop_sequences: str | list[str], tokenizer: transformers.PreTrainedTokenizer):
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        self.stop_sequences = stop_sequences
        self.tokenizer: transformers.PreTrainedTokenizer = tokenizer

    def __call__(self, input_ids: torch.Tensor, scores: t.Any, **attrs: t.Any) -> bool:
        decoded_output = self.tokenizer.decode(input_ids.tolist()[0])
        return any(decoded_output.endswith(stop_sequence) for stop_sequence in self.stop_sequences)


class StopOnTokens(transformers.StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: t.Any) -> bool:
        stop_ids = {50278, 50279, 50277, 1, 0}
        return t.cast(int, input_ids[0][-1]) in stop_ids
