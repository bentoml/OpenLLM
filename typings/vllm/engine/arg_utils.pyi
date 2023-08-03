import argparse
from typing import Optional
from typing import Tuple

from vllm.config import CacheConfig as CacheConfig
from vllm.config import ModelConfig as ModelConfig
from vllm.config import ParallelConfig as ParallelConfig
from vllm.config import SchedulerConfig as SchedulerConfig

class EngineArgs:
    model: str
    tokenizer: Optional[str]
    tokenizer_mode: str
    trust_remote_code: bool
    download_dir: Optional[str]
    use_np_weights: bool
    use_dummy_weights: bool
    dtype: str
    seed: int
    worker_use_ray: bool
    pipeline_parallel_size: int
    tensor_parallel_size: int
    block_size: int
    swap_space: int
    gpu_memory_utilization: float
    max_num_batched_tokens: int
    max_num_seqs: int
    disable_log_stats: bool
    def __post_init__(self) -> None: ...
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: ...
    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> EngineArgs: ...
    def create_engine_configs(self) -> Tuple[ModelConfig, CacheConfig, ParallelConfig, SchedulerConfig]: ...
    def __init__(self, model, tokenizer, tokenizer_mode, trust_remote_code, download_dir, use_np_weights, use_dummy_weights, dtype, seed, worker_use_ray, pipeline_parallel_size, tensor_parallel_size, block_size, swap_space, gpu_memory_utilization, max_num_batched_tokens, max_num_seqs, disable_log_stats) -> None: ...

class AsyncEngineArgs(EngineArgs):
    engine_use_ray: bool
    disable_log_requests: bool
    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser: ...
    def __init__(self, model, tokenizer, tokenizer_mode, trust_remote_code, download_dir, use_np_weights, use_dummy_weights, dtype, seed, worker_use_ray, pipeline_parallel_size, tensor_parallel_size, block_size, swap_space, gpu_memory_utilization, max_num_batched_tokens, max_num_seqs, disable_log_stats, engine_use_ray, disable_log_requests) -> None: ...
