from _typeshed import Incomplete
from transformers.configuration_utils import PretrainedConfig

class BaiChuanConfig(PretrainedConfig):
    model_type: str
    keys_to_ignore_at_inference: Incomplete
    vocab_size: Incomplete
    max_position_embeddings: Incomplete
    hidden_size: Incomplete
    intermediate_size: Incomplete
    num_hidden_layers: Incomplete
    num_attention_heads: Incomplete
    hidden_act: Incomplete
    initializer_range: Incomplete
    rms_norm_eps: Incomplete
    use_cache: Incomplete
    def __init__(self, vocab_size: int = ..., hidden_size: int = ..., intermediate_size: int = ..., num_hidden_layers: int = ..., num_attention_heads: int = ..., hidden_act: str = ..., max_position_embeddings: int = ..., initializer_range: float = ..., rms_norm_eps: float = ..., use_cache: bool = ..., pad_token_id: int = ..., bos_token_id: int = ..., eos_token_id: int = ..., tie_word_embeddings: bool = ..., **kwargs) -> None: ...
