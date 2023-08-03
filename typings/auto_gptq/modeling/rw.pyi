from _typeshed import Incomplete

from ._base import *

class RWGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type: str
    layers_block_name: str
    outside_layer_modules: Incomplete
    inside_layer_modules: Incomplete
