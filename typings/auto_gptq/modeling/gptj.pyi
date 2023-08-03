from _typeshed import Incomplete

from ._base import *
from ..nn_modules.fused_gptj_attn import FusedGPTJAttentionForQuantizedModel

class GPTJGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type: str
    layers_block_name: str
    outside_layer_modules: Incomplete
    inside_layer_modules: Incomplete
    fused_attn_module_type = FusedGPTJAttentionForQuantizedModel
