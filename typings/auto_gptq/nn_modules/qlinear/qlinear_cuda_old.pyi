from _typeshed import Incomplete
from torch import nn

class QuantLinear(nn.Module):
    infeatures: Incomplete
    outfeatures: Incomplete
    bits: Incomplete
    group_size: Incomplete
    maxq: Incomplete
    bias: Incomplete
    half_indim: Incomplete
    use_cuda_fp16: Incomplete
    wf: Incomplete
    kernel_switch_threshold: Incomplete
    autogptq_cuda_available: Incomplete
    autogptq_cuda: Incomplete
    trainable: Incomplete
    def __init__(self, bits, group_size, infeatures, outfeatures, bias, use_cuda_fp16: bool = ..., kernel_switch_threshold: int = ..., trainable: bool = ...) -> None: ...
    scales: Incomplete
    qweight: Incomplete
    qzeros: Incomplete
    def pack(self, linear, scales, zeros, g_idx) -> None: ...
    def forward(self, x): ...
