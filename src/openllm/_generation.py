"""Generation utilities to be reused throughout."""
from __future__ import annotations

import transformers


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
