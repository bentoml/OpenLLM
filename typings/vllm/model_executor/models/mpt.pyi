from typing import Dict
from typing import List
from typing import Optional

import torch
from _typeshed import Incomplete
from torch import nn
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
from vllm.transformers_utils.configs.mpt import MPTConfig as MPTConfig

KVCache: Incomplete

class MPTAttention(nn.Module):
    d_model: Incomplete
    total_num_heads: Incomplete
    clip_qkv: Incomplete
    qk_ln: Incomplete
    alibi_bias_max: Incomplete
    qkv_proj: Incomplete
    q_ln: Incomplete
    k_ln: Incomplete
    out_proj: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    attn: Incomplete
    def __init__(self, config: MPTConfig) -> None: ...
    def forward(self, position_ids: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata, cache_event: Optional[torch.cuda.Event]) -> torch.Tensor: ...

class MPTMLP(nn.Module):
    up_proj: Incomplete
    act: Incomplete
    down_proj: Incomplete
    def __init__(self, config: MPTConfig) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

class MPTBlock(nn.Module):
    norm_1: Incomplete
    attn: Incomplete
    norm_2: Incomplete
    ffn: Incomplete
    def __init__(self, config: MPTConfig) -> None: ...
    def forward(self, position_ids: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata, cache_event: Optional[torch.cuda.Event]) -> torch.Tensor: ...

class MPTModel(nn.Module):
    wte: Incomplete
    blocks: Incomplete
    norm_f: Incomplete
    def __init__(self, config: MPTConfig) -> None: ...
    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata, cache_events: Optional[List[torch.cuda.Event]]) -> torch.Tensor: ...

class MPTForCausalLM(nn.Module):
    config: Incomplete
    transformer: Incomplete
    lm_head_weight: Incomplete
    sampler: Incomplete
    def __init__(self, config: MPTConfig) -> None: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata, cache_events: Optional[List[torch.cuda.Event]]) -> Dict[int, SequenceOutputs]: ...
    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str] = ..., use_np_cache: bool = ...): ...
