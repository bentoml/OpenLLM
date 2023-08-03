from typing import Dict
from typing import Optional

import torch
from _typeshed import Incomplete
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata as InputMetadata
from vllm.model_executor.parallel_utils.tensor_parallel import (
    gather_from_tensor_model_parallel_region as gather_from_tensor_model_parallel_region,
)
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.sequence import SequenceOutputs as SequenceOutputs

class Sampler(nn.Module):
    vocab_size: Incomplete
    def __init__(self, vocab_size: int) -> None: ...
    def forward(self, embedding: torch.Tensor, hidden_states: torch.Tensor, input_metadata: InputMetadata, embedding_bias: Optional[torch.Tensor] = ...) -> Dict[int, SequenceOutputs]: ...
