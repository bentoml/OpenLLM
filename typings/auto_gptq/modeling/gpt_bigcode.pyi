from _typeshed import Incomplete
from auto_gptq.modeling import BaseGPTQForCausalLM

class GPTBigCodeGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type: str
    layers_block_name: str
    outside_layer_modules: Incomplete
    inside_layer_modules: Incomplete
