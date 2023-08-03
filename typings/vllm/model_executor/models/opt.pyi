from typing import Dict
from typing import List
from typing import Optional

import torch
from _typeshed import Incomplete
from torch import nn
from transformers import OPTConfig as OPTConfig
from vllm.model_executor.input_metadata import InputMetadata as InputMetadata
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import PagedAttention as PagedAttention
from vllm.model_executor.layers.sampler import Sampler as Sampler
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank as get_tensor_model_parallel_rank,
)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_world_size as get_tensor_model_parallel_world_size,
)
from vllm.model_executor.parallel_utils.tensor_parallel import ColumnParallelLinear as ColumnParallelLinear
from vllm.model_executor.parallel_utils.tensor_parallel import RowParallelLinear as RowParallelLinear
from vllm.model_executor.parallel_utils.tensor_parallel import VocabParallelEmbedding as VocabParallelEmbedding
from vllm.model_executor.weight_utils import hf_model_weights_iterator as hf_model_weights_iterator
from vllm.model_executor.weight_utils import load_tensor_parallel_weights as load_tensor_parallel_weights
from vllm.sequence import SequenceOutputs as SequenceOutputs

KVCache: Incomplete

class OPTLearnedPositionalEmbedding(nn.Embedding):
    offset: int
    def __init__(self, num_embeddings: int, embedding_dim: int) -> None: ...
    def forward(self, positions: torch.Tensor): ...

class OPTAttention(nn.Module):
    embed_dim: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    scaling: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    attn: Incomplete
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = ...) -> None: ...
    def forward(self, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata, cache_event: Optional[torch.cuda.Event]) -> torch.Tensor: ...

class OPTDecoderLayer(nn.Module):
    config: Incomplete
    embed_dim: Incomplete
    self_attn: Incomplete
    do_layer_norm_before: Incomplete
    activation_fn: Incomplete
    self_attn_layer_norm: Incomplete
    fc1: Incomplete
    fc2: Incomplete
    final_layer_norm: Incomplete
    def __init__(self, config: OPTConfig) -> None: ...
    def forward(self, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata, cache_event: Optional[torch.cuda.Event]) -> torch.Tensor: ...

class OPTDecoder(nn.Module):
    config: Incomplete
    padding_idx: Incomplete
    max_target_positions: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    embed_positions: Incomplete
    project_out: Incomplete
    project_in: Incomplete
    final_layer_norm: Incomplete
    layers: Incomplete
    def __init__(self, config: OPTConfig) -> None: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata, cache_events: Optional[List[torch.cuda.Event]]) -> torch.Tensor: ...

class OPTModel(nn.Module):
    decoder: Incomplete
    def __init__(self, config: OPTConfig) -> None: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata, cache_events: Optional[List[torch.cuda.Event]]) -> torch.Tensor: ...

class OPTForCausalLM(nn.Module):
    config: Incomplete
    model: Incomplete
    lm_head_weight: Incomplete
    sampler: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata, cache_events: Optional[List[torch.cuda.Event]]) -> Dict[int, SequenceOutputs]: ...
    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str] = ..., use_np_cache: bool = ...): ...
