from typing import Dict
from typing import List
from typing import Optional

import torch
from _typeshed import Incomplete
from torch import nn
from transformers import BloomConfig as BloomConfig
from vllm.model_executor.input_metadata import InputMetadata as InputMetadata
from vllm.model_executor.layers.activation import get_act_fn as get_act_fn
from vllm.model_executor.layers.attention import PagedAttentionWithALiBi as PagedAttentionWithALiBi
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

class BloomAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    head_dim: Incomplete
    num_heads: Incomplete
    query_key_value: Incomplete
    dense: Incomplete
    attn: Incomplete
    def __init__(self, config: BloomConfig) -> None: ...
    def forward(self, position_ids: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata, cache_event: Optional[torch.cuda.Event]) -> torch.Tensor: ...

class BloomMLP(nn.Module):
    dense_h_to_4h: Incomplete
    act: Incomplete
    dense_4h_to_h: Incomplete
    def __init__(self, config: BloomConfig) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class BloomBlock(nn.Module):
    input_layernorm: Incomplete
    self_attention: Incomplete
    post_attention_layernorm: Incomplete
    mlp: Incomplete
    apply_residual_connection_post_layernorm: Incomplete
    def __init__(self, config: BloomConfig) -> None: ...
    def forward(self, position_ids: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata, cache_event: Optional[torch.cuda.Event]) -> torch.Tensor: ...

class BloomModel(nn.Module):
    embed_dim: Incomplete
    word_embeddings: Incomplete
    word_embeddings_layernorm: Incomplete
    h: Incomplete
    ln_f: Incomplete
    def __init__(self, config: BloomConfig) -> None: ...
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata, cache_events: Optional[List[torch.cuda.Event]]) -> torch.Tensor: ...

class BloomForCausalLM(nn.Module):
    config: Incomplete
    transformer: Incomplete
    lm_head_weight: Incomplete
    sampler: Incomplete
    def __init__(self, config: BloomConfig) -> None: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata, cache_events: Optional[List[torch.cuda.Event]]) -> Dict[int, SequenceOutputs]: ...
    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str] = ..., use_np_cache: bool = ...): ...
