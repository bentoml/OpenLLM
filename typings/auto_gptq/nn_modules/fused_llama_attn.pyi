from _typeshed import Incomplete

from ._fused_base import FusedBaseAttentionModule

class FusedLlamaAttentionForQuantizedModel(FusedBaseAttentionModule):
    hidden_size: Incomplete
    num_heads: Incomplete
    head_dim: Incomplete
    qkv_proj: Incomplete
    o_proj: Incomplete
    rotary_emb: Incomplete
    def __init__(self, hidden_size, num_heads, qkv_proj, o_proj, rotary_emb) -> None: ...
    def forward(self, hidden_states, past_key_value: Incomplete | None = ..., attention_mask: Incomplete | None = ..., position_ids: Incomplete | None = ..., output_attentions: bool = ..., use_cache: bool = ..., **kwargs): ...
    @classmethod
    def inject_to_model(cls, model, use_triton: bool = ..., group_size: int = ..., use_cuda_fp16: bool = ..., desc_act: bool = ..., trainable: bool = ..., **kwargs) -> None: ...
