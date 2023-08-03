from _typeshed import Incomplete

class GPTQ:
    layer: Incomplete
    dev: Incomplete
    rows: Incomplete
    columns: Incomplete
    H: Incomplete
    nsamples: int
    quantizer: Incomplete
    def __init__(self, layer) -> None: ...
    inp1: Incomplete
    out1: Incomplete
    def add_batch(self, inp, out) -> None: ...
    def fasterquant(self, blocksize: int = ..., percdamp: float = ..., group_size: int = ..., actorder: bool = ...): ...
    Losses: Incomplete
    Trace: Incomplete
    def free(self) -> None: ...
