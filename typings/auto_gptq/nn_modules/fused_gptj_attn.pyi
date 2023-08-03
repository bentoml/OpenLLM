from typing import *

import torch
from _typeshed import Incomplete

from ._fused_base import FusedBaseAttentionModule

class FusedGPTJAttentionForQuantizedModel(FusedBaseAttentionModule):
    attn_dropout: Incomplete
    attn_dropout_p: Incomplete
    resid_dropout: Incomplete
    embed_dim: Incomplete
    num_attention_heads: Incomplete
    head_dim: Incomplete
    scale_attn: Incomplete
    qkv_proj: Incomplete
    out_proj: Incomplete
    rotary_dim: Incomplete
    def __init__(self, config) -> None: ...
    def forward(self, hidden_states: torch.FloatTensor, layer_past: Optional[Tuple[torch.Tensor]] = ..., attention_mask: Optional[torch.FloatTensor] = ..., position_ids: Optional[torch.LongTensor] = ..., head_mask: Optional[torch.FloatTensor] = ..., use_cache: Optional[bool] = ..., output_attentions: Optional[bool] = ...) -> Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]]]: ...
    @classmethod
    def inject_to_model(cls, model, use_triton: bool = ..., group_size: int = ..., use_cuda_fp16: bool = ..., desc_act: bool = ..., trainable: bool = ..., **kwargs) -> None: ...
