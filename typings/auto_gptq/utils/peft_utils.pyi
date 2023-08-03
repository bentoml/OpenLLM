from typing import List
from typing import Optional

import torch
from _typeshed import Incomplete
from peft import PeftConfig
from peft.tuners.adalora import AdaLoraConfig
from peft.tuners.adalora import AdaLoraLayer
from peft.tuners.adalora import AdaLoraModel
from peft.tuners.lora import LoraConfig
from peft.tuners.lora import LoraLayer
from peft.tuners.lora import LoraModel

from ..modeling._base import BaseGPTQForCausalLM

class GPTQLoraConfig(LoraConfig):
    injected_fused_attention: bool
    injected_fused_mlp: bool

class GPTQLoraLinear(torch.nn.Linear, LoraLayer):
    linear_module: Incomplete
    weight: Incomplete
    bias: Incomplete
    fan_in_fan_out: Incomplete
    active_adapter: Incomplete
    def __init__(self, adapter_name: str, linear_module: torch.nn.Linear, r: int = ..., lora_alpha: int = ..., lora_dropout: float = ..., fan_in_fan_out: bool = ..., **kwargs) -> None: ...
    def reset_lora_parameters(self, adapter_name) -> None: ...
    def merge(self) -> None: ...
    def unmerge(self) -> None: ...
    def forward(self, x: torch.Tensor): ...

class GPTQLoraModel(LoraModel):
    def merge_adapter(self) -> None: ...
    def unmerge_adapter(self) -> None: ...
    def merge_and_unload(self) -> None: ...

class GPTQAdaLoraConfig(AdaLoraConfig):
    injected_fused_attention: bool
    injected_fused_mlp: bool

class GPTQSVDLinear(torch.nn.Linear, AdaLoraLayer):
    linear_module: Incomplete
    weight: Incomplete
    bias: Incomplete
    fan_in_fan_out: Incomplete
    active_adapter: Incomplete
    def __init__(self, adapter_name: str, linear_module: torch.nn.Linear, r: int = ..., lora_alpha: int = ..., lora_dropout: float = ..., fan_in_fan_out: bool = ..., **kwargs) -> None: ...
    def merge(self) -> None: ...
    def unmerge(self) -> None: ...
    def forward(self, x: torch.Tensor): ...

class GPTQAdaLoraModel(AdaLoraModel):
    def merge_adapter(self) -> None: ...
    def unmerge_adapter(self) -> None: ...
    def merge_and_unload(self) -> None: ...

def find_all_linear_names(model: BaseGPTQForCausalLM, ignore: Optional[List[str]] = ..., ignore_lm_head: bool = ...): ...
def get_gptq_peft_model(model: BaseGPTQForCausalLM, peft_config: PeftConfig = ..., model_id: str = ..., adapter_name: str = ..., auto_find_all_linears: bool = ..., train_mode: bool = ...): ...
