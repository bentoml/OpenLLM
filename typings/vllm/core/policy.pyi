from typing import List

from vllm.sequence import SequenceGroup as SequenceGroup

class Policy:
    def get_priority(self, now: float, seq_group: SequenceGroup) -> float: ...
    def sort_by_priority(self, now: float, seq_groups: List[SequenceGroup]) -> List[SequenceGroup]: ...

class FCFS(Policy):
    def get_priority(self, now: float, seq_group: SequenceGroup) -> float: ...

class PolicyFactory:
    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> Policy: ...
