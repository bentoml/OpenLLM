from _typeshed import Incomplete

from ._base import *
from ..nn_modules.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
from ..nn_modules.fused_llama_mlp import FusedLlamaMLPForQuantizedModel

class LlamaGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type: str
    layers_block_name: str
    outside_layer_modules: Incomplete
    inside_layer_modules: Incomplete
    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel
