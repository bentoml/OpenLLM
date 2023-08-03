from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from _typeshed import Incomplete
from vllm.config import CacheConfig as CacheConfig
from vllm.config import ModelConfig as ModelConfig
from vllm.config import ParallelConfig as ParallelConfig
from vllm.config import SchedulerConfig as SchedulerConfig
from vllm.model_executor import InputMetadata as InputMetadata
from vllm.model_executor import get_model as get_model
from vllm.model_executor import set_random_seed as set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_all_reduce_launcher as initialize_all_reduce_launcher,
)
from vllm.model_executor.parallel_utils.parallel_state import initialize_model_parallel as initialize_model_parallel
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.sequence import SequenceData as SequenceData
from vllm.sequence import SequenceGroupMetadata as SequenceGroupMetadata
from vllm.sequence import SequenceOutputs as SequenceOutputs
from vllm.utils import get_gpu_memory as get_gpu_memory
from vllm.worker.cache_engine import CacheEngine as CacheEngine

class Worker:
    model_config: Incomplete
    parallel_config: Incomplete
    scheduler_config: Incomplete
    rank: Incomplete
    distributed_init_method: Incomplete
    cache_config: Incomplete
    block_size: Incomplete
    cache_engine: Incomplete
    cache_events: Incomplete
    gpu_cache: Incomplete
    def __init__(self, model_config: ModelConfig, parallel_config: ParallelConfig, scheduler_config: SchedulerConfig, rank: Optional[int] = ..., distributed_init_method: Optional[str] = ...) -> None: ...
    device: Incomplete
    model: Incomplete
    def init_model(self) -> None: ...
    def profile_num_available_blocks(self, block_size: int, gpu_memory_utilization: float, cpu_swap_space: int) -> Tuple[int, int]: ...
    def init_cache_engine(self, cache_config: CacheConfig) -> None: ...
    def execute_model(self, seq_group_metadata_list: List[SequenceGroupMetadata], blocks_to_swap_in: Dict[int, int], blocks_to_swap_out: Dict[int, int], blocks_to_copy: Dict[int, List[int]]) -> Dict[int, SequenceOutputs]: ...
