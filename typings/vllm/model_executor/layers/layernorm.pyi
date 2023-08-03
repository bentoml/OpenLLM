import torch
from _typeshed import Incomplete
from torch import nn
from vllm import layernorm_ops as layernorm_ops

class RMSNorm(nn.Module):
    weight: Incomplete
    variance_epsilon: Incomplete
    def __init__(self, hidden_size: int, eps: float = ...) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
