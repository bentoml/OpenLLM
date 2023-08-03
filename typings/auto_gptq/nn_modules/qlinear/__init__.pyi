from _typeshed import Incomplete
from torch import nn

class GeneralQuantLinear(nn.Linear):
    infeatures: Incomplete
    outfeatures: Incomplete
    bits: Incomplete
    group_size: Incomplete
    maxq: Incomplete
    qweight: Incomplete
    qzeros: Incomplete
    scales: Incomplete
    g_idx: Incomplete
    wf: Incomplete
    kernel_switch_threshold: Incomplete
    autogptq_cuda_available: Incomplete
    trainable: Incomplete
    forward: Incomplete
    def __init__(self, quant_linear_module) -> None: ...
    @classmethod
    def inject_to_model(cls, model, target_module_type) -> None: ...
