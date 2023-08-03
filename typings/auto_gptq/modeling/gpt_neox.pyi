from _typeshed import Incomplete

from ._base import *

class GPTNeoXGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type: str
    layers_block_name: str
    outside_layer_modules: Incomplete
    inside_layer_modules: Incomplete
    lm_head_name: str
