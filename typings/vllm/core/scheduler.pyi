import enum
from typing import Dict
from typing import List
from typing import Tuple

from _typeshed import Incomplete
from vllm.config import CacheConfig as CacheConfig
from vllm.config import SchedulerConfig as SchedulerConfig
from vllm.core.block_manager import BlockSpaceManager as BlockSpaceManager
from vllm.core.policy import PolicyFactory as PolicyFactory
from vllm.logger import init_logger as init_logger
from vllm.sequence import Sequence as Sequence
from vllm.sequence import SequenceData as SequenceData
from vllm.sequence import SequenceGroup as SequenceGroup
from vllm.sequence import SequenceGroupMetadata as SequenceGroupMetadata
from vllm.sequence import SequenceOutputs as SequenceOutputs
from vllm.sequence import SequenceStatus as SequenceStatus

logger: Incomplete

class PreemptionMode(enum.Enum):
    SWAP: Incomplete
    RECOMPUTE: Incomplete

class SchedulerOutputs:
    blocks_to_swap_in: Incomplete
    blocks_to_swap_out: Incomplete
    blocks_to_copy: Incomplete
    def __init__(self, blocks_to_swap_in: Dict[int, int], blocks_to_swap_out: Dict[int, int], blocks_to_copy: Dict[int, List[int]]) -> None: ...
    def is_empty(self) -> bool: ...

class Scheduler:
    scheduler_config: Incomplete
    cache_config: Incomplete
    log_stats: Incomplete
    policy: Incomplete
    block_manager: Incomplete
    waiting: Incomplete
    running: Incomplete
    swapped: Incomplete
    last_logging_time: float
    num_input_tokens: Incomplete
    def __init__(self, scheduler_config: SchedulerConfig, cache_config: CacheConfig, log_stats: bool) -> None: ...
    def add_seq_group(self, seq_group: SequenceGroup) -> None: ...
    def abort_seq_group(self, request_id: str) -> None: ...
    def has_unfinished_seqs(self) -> bool: ...
    def get_num_unfinished_seq_groups(self) -> int: ...
    def schedule(self) -> Tuple[List[SequenceGroupMetadata], SchedulerOutputs, List[SequenceGroup]]: ...
    def update(self, seq_outputs: Dict[int, SequenceOutputs]) -> List[SequenceGroup]: ...
    def free_seq(self, seq: Sequence, finish_status: SequenceStatus) -> None: ...
    def free_finished_seq_groups(self) -> None: ...
