from _typeshed import Incomplete

from ._fused_base import FusedBaseMLPModule

class FusedLlamaMLPForQuantizedModel(FusedBaseMLPModule):
    infeatures: Incomplete
    intermediate_size: Incomplete
    outfeatures: Incomplete
    bits: Incomplete
    maxq: Incomplete
    gate_proj: Incomplete
    up_proj: Incomplete
    down_proj: Incomplete
    def __init__(self, gate_proj, down_proj, up_proj) -> None: ...
    def forward(self, x): ...
    def triton_llama_mlp(self, x): ...
    @classmethod
    def inject_to_model(cls, model, use_triton: bool = ..., **kwargs) -> None: ...
    @classmethod
    def warmup(cls, model, transpose: bool = ..., seqlen: int = ...) -> None: ...
