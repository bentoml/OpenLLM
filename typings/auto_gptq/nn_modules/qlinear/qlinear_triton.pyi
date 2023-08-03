from _typeshed import Incomplete
from torch import nn

from ..triton_utils.mixin import TritonModuleMixin

class QuantLinear(nn.Module, TritonModuleMixin):
    infeatures: Incomplete
    outfeatures: Incomplete
    bits: Incomplete
    group_size: Incomplete
    maxq: Incomplete
    bias: Incomplete
    trainable: Incomplete
    def __init__(self, bits, group_size, infeatures, outfeatures, bias, trainable: bool = ...) -> None: ...
    g_idx: Incomplete
    scales: Incomplete
    qweight: Incomplete
    qzeros: Incomplete
    def pack(self, linear, scales, zeros, g_idx: Incomplete | None = ...) -> None: ...
    def forward(self, x): ...
    @classmethod
    def warmup(cls, model, transpose: bool = ..., seqlen: int = ...) -> None: ...
