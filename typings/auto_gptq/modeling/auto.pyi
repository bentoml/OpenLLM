from typing import Dict
from typing import Optional
from typing import Union

from ._base import BaseGPTQForCausalLM
from ._base import BaseQuantizeConfig

class AutoGPTQForCausalLM:
    def __init__(self) -> None: ...
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, quantize_config: BaseQuantizeConfig, max_memory: Optional[dict] = ..., trust_remote_code: bool = ..., **model_init_kwargs) -> BaseGPTQForCausalLM: ...
    @classmethod
    def from_quantized(cls, model_name_or_path: Optional[str], device_map: Optional[Union[str, Dict[str, Union[str, int]]]] = ..., max_memory: Optional[dict] = ..., device: Optional[Union[str, int]] = ..., low_cpu_mem_usage: bool = ..., use_triton: bool = ..., inject_fused_attention: bool = ..., inject_fused_mlp: bool = ..., use_cuda_fp16: bool = ..., quantize_config: Optional[BaseQuantizeConfig] = ..., model_basename: Optional[str] = ..., use_safetensors: bool = ..., trust_remote_code: bool = ..., warmup_triton: bool = ..., trainable: bool = ..., **kwargs) -> BaseGPTQForCausalLM: ...
