from typing import Any
from typing import Dict
from typing import Optional
from typing import Union

from _typeshed import Incomplete
from transformers import PretrainedConfig

class MPTConfig(PretrainedConfig):
    model_type: str
    attribute_map: Incomplete
    d_model: Incomplete
    n_heads: Incomplete
    n_layers: Incomplete
    expansion_ratio: Incomplete
    max_seq_len: Incomplete
    vocab_size: Incomplete
    resid_pdrop: Incomplete
    emb_pdrop: Incomplete
    learned_pos_emb: Incomplete
    attn_config: Incomplete
    init_device: Incomplete
    logit_scale: Incomplete
    no_bias: Incomplete
    verbose: Incomplete
    embedding_fraction: Incomplete
    norm_type: Incomplete
    use_cache: Incomplete
    def __init__(self, d_model: int = ..., n_heads: int = ..., n_layers: int = ..., expansion_ratio: int = ..., max_seq_len: int = ..., vocab_size: int = ..., resid_pdrop: float = ..., emb_pdrop: float = ..., learned_pos_emb: bool = ..., attn_config: Optional[Dict[str, Any]] = ..., init_device: str = ..., logit_scale: Optional[Union[float, str]] = ..., no_bias: bool = ..., verbose: int = ..., embedding_fraction: float = ..., norm_type: str = ..., use_cache: bool = ..., **kwargs) -> None: ...
