from transformers import PretrainedConfig as PretrainedConfig
from vllm.transformers_utils.configs import *

def get_config(model: str, trust_remote_code: bool) -> PretrainedConfig: ...
