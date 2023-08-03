from typing import List
from typing import Optional

from _typeshed import Incomplete
from ray.util.placement_group import PlacementGroup as PlacementGroup
from vllm.config import CacheConfig as CacheConfig
from vllm.config import ModelConfig as ModelConfig
from vllm.config import ParallelConfig as ParallelConfig
from vllm.config import SchedulerConfig as SchedulerConfig
from vllm.core.scheduler import Scheduler as Scheduler
from vllm.engine.arg_utils import EngineArgs as EngineArgs
from vllm.engine.ray_utils import RayWorker as RayWorker
from vllm.engine.ray_utils import initialize_cluster as initialize_cluster
from vllm.engine.ray_utils import ray as ray
from vllm.logger import init_logger as init_logger
from vllm.outputs import RequestOutput as RequestOutput
from vllm.sampling_params import SamplingParams as SamplingParams
from vllm.sequence import Sequence as Sequence
from vllm.sequence import SequenceGroup as SequenceGroup
from vllm.sequence import SequenceStatus as SequenceStatus
from vllm.transformers_utils.tokenizer import detokenize_incrementally as detokenize_incrementally
from vllm.transformers_utils.tokenizer import get_tokenizer as get_tokenizer
from vllm.utils import Counter as Counter

logger: Incomplete

class LLMEngine:
    model_config: Incomplete
    cache_config: Incomplete
    parallel_config: Incomplete
    scheduler_config: Incomplete
    log_stats: Incomplete
    tokenizer: Incomplete
    seq_counter: Incomplete
    scheduler: Incomplete
    def __init__(self, model_config: ModelConfig, cache_config: CacheConfig, parallel_config: ParallelConfig, scheduler_config: SchedulerConfig, distributed_init_method: str, placement_group: Optional[PlacementGroup], log_stats: bool) -> None: ...
    @classmethod
    def from_engine_args(cls, engine_args: EngineArgs) -> LLMEngine: ...
    def add_request(self, request_id: str, prompt: Optional[str], sampling_params: SamplingParams, prompt_token_ids: Optional[List[int]] = ..., arrival_time: Optional[float] = ...) -> None: ...
    def abort_request(self, request_id: str) -> None: ...
    def get_model_config(self) -> ModelConfig: ...
    def get_num_unfinished_requests(self) -> int: ...
    def has_unfinished_requests(self) -> bool: ...
    def step(self) -> List[RequestOutput]: ...
