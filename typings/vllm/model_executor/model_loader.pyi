from torch import nn
from transformers import PretrainedConfig as PretrainedConfig
from vllm.config import ModelConfig as ModelConfig
from vllm.model_executor.models import *
from vllm.model_executor.weight_utils import initialize_dummy_weights as initialize_dummy_weights

def get_model(model_config: ModelConfig) -> nn.Module: ...
