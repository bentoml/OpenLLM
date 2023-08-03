import torch
from torch import nn
from vllm import activation_ops as activation_ops

def get_act_fn(act_fn: str) -> nn.Module: ...

class SiluAndMul(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
