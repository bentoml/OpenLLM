from typing import Optional

from _typeshed import Incomplete
from transformers import PretrainedConfig as PretrainedConfig
from vllm.logger import init_logger as init_logger
from vllm.transformers_utils.config import get_config as get_config
from vllm.utils import get_cpu_memory as get_cpu_memory

logger: Incomplete

class ModelConfig:
    model: Incomplete
    tokenizer: Incomplete
    tokenizer_mode: Incomplete
    trust_remote_code: Incomplete
    download_dir: Incomplete
    use_np_weights: Incomplete
    use_dummy_weights: Incomplete
    seed: Incomplete
    hf_config: Incomplete
    dtype: Incomplete
    def __init__(self, model: str, tokenizer: str, tokenizer_mode: str, trust_remote_code: bool, download_dir: Optional[str], use_np_weights: bool, use_dummy_weights: bool, dtype: str, seed: int) -> None: ...
    def verify_with_parallel_config(self, parallel_config: ParallelConfig) -> None: ...
    def get_hidden_size(self) -> int: ...
    def get_head_size(self) -> int: ...
    def get_num_heads(self, parallel_config: ParallelConfig) -> int: ...
    def get_max_model_len(self) -> int: ...
    def get_num_layers(self, parallel_config: ParallelConfig) -> int: ...

class CacheConfig:
    block_size: Incomplete
    gpu_memory_utilization: Incomplete
    swap_space_bytes: Incomplete
    num_gpu_blocks: Incomplete
    num_cpu_blocks: Incomplete
    def __init__(self, block_size: int, gpu_memory_utilization: float, swap_space: int) -> None: ...
    def verify_with_parallel_config(self, parallel_config: ParallelConfig) -> None: ...

class ParallelConfig:
    pipeline_parallel_size: Incomplete
    tensor_parallel_size: Incomplete
    worker_use_ray: Incomplete
    world_size: Incomplete
    def __init__(self, pipeline_parallel_size: int, tensor_parallel_size: int, worker_use_ray: bool) -> None: ...

class SchedulerConfig:
    max_num_batched_tokens: Incomplete
    max_num_seqs: Incomplete
    max_model_len: Incomplete
    def __init__(self, max_num_batched_tokens: int, max_num_seqs: int, max_model_len: int) -> None: ...
