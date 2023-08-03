from typing import Dict
from typing import List
from typing import Optional

from _typeshed import Incomplete
from vllm.sequence import SequenceGroup as SequenceGroup
from vllm.sequence import SequenceStatus as SequenceStatus

class CompletionOutput:
    index: Incomplete
    text: Incomplete
    token_ids: Incomplete
    cumulative_logprob: Incomplete
    logprobs: Incomplete
    finish_reason: Incomplete
    def __init__(self, index: int, text: str, token_ids: List[int], cumulative_logprob: float, logprobs: Optional[List[Dict[int, float]]], finish_reason: Optional[str] = ...) -> None: ...
    def finished(self) -> bool: ...

class RequestOutput:
    request_id: Incomplete
    prompt: Incomplete
    prompt_token_ids: Incomplete
    outputs: Incomplete
    finished: Incomplete
    def __init__(self, request_id: str, prompt: str, prompt_token_ids: List[int], outputs: List[CompletionOutput], finished: bool) -> None: ...
    @classmethod
    def from_seq_group(cls, seq_group: SequenceGroup) -> RequestOutput: ...
