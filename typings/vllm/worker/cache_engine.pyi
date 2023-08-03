from typing import Dict
from typing import List
from typing import Tuple

from _typeshed import Incomplete
from vllm import cache_ops as cache_ops
from vllm.config import CacheConfig as CacheConfig
from vllm.config import ModelConfig as ModelConfig
from vllm.config import ParallelConfig as ParallelConfig
from vllm.logger import init_logger as init_logger
from vllm.utils import in_wsl as in_wsl

logger: Incomplete
KVCache: Incomplete

class CacheEngine:
    cache_config: Incomplete
    model_config: Incomplete
    parallel_config: Incomplete
    head_size: Incomplete
    num_layers: Incomplete
    num_heads: Incomplete
    dtype: Incomplete
    block_size: Incomplete
    num_gpu_blocks: Incomplete
    num_cpu_blocks: Incomplete
    gpu_cache: Incomplete
    cpu_cache: Incomplete
    cache_stream: Incomplete
    events: Incomplete
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig, parallel_config: ParallelConfig) -> None: ...
    def get_key_block_shape(self) -> Tuple[int, int, int, int]: ...
    def get_value_block_shape(self) -> Tuple[int, int, int]: ...
    def allocate_gpu_cache(self) -> List[KVCache]: ...
    def allocate_cpu_cache(self) -> List[KVCache]: ...
    def swap_in(self, src_to_dst: Dict[int, int]) -> None: ...
    def swap_out(self, src_to_dst: Dict[int, int]) -> None: ...
    def copy(self, src_to_dsts: Dict[int, List[int]]) -> None: ...
    @staticmethod
    def get_cache_block_size(block_size: int, model_config: ModelConfig, parallel_config: ParallelConfig) -> int: ...
