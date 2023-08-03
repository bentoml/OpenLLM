from typing import Dict
from typing import List
from typing import Optional

import torch
from _typeshed import Incomplete
from torch import nn
from transformers import LlamaConfig as LlamaConfig
from vllm.model_executor.input_metadata import InputMetadata as InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul as SiluAndMul
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE as PagedAttentionWithRoPE
from vllm.model_executor.layers.layernorm import RMSNorm as RMSNorm
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

class LlamaMLP(nn.Module):
    gate_up_proj: Incomplete
    down_proj: Incomplete
    act_fn: Incomplete
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None: ...
    def forward(self, x): ...

class LlamaAttention(nn.Module):
    hidden_size: Incomplete
    total_num_heads: Incomplete
    num_heads: Incomplete
    total_num_kv_heads: Incomplete
    num_kv_heads: Incomplete
    head_dim: Incomplete
    q_size: Incomplete
    kv_size: Incomplete
    scaling: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    attn: Incomplete
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int) -> None: ...
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata, cache_event: Optional[torch.cuda.Event]) -> torch.Tensor: ...

class LlamaDecoderLayer(nn.Module):
    hidden_size: Incomplete
    self_attn: Incomplete
    mlp: Incomplete
    input_layernorm: Incomplete
    post_attention_layernorm: Incomplete
    def __init__(self, config: LlamaConfig) -> None: ...
    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata, cache_event: Optional[torch.cuda.Event]) -> torch.Tensor: ...

class LlamaModel(nn.Module):
    config: Incomplete
    padding_idx: Incomplete
    vocab_size: Incomplete
    embed_tokens: Incomplete
    layers: Incomplete
    norm: Incomplete
    def __init__(self, config: LlamaConfig) -> None: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata, cache_events: Optional[List[torch.cuda.Event]]) -> torch.Tensor: ...

class LlamaForCausalLM(nn.Module):
    config: Incomplete
    model: Incomplete
    lm_head: Incomplete
    sampler: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata, cache_events: Optional[List[torch.cuda.Event]]) -> Dict[int, SequenceOutputs]: ...
    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str] = ..., use_np_cache: bool = ...): ...
