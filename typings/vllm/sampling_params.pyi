from typing import List
from typing import Optional
from typing import Union

from _typeshed import Incomplete

class SamplingParams:
    n: Incomplete
    best_of: Incomplete
    presence_penalty: Incomplete
    frequency_penalty: Incomplete
    temperature: Incomplete
    top_p: Incomplete
    top_k: Incomplete
    use_beam_search: Incomplete
    stop: Incomplete
    ignore_eos: Incomplete
    max_tokens: Incomplete
    logprobs: Incomplete
    def __init__(self, n: int = ..., best_of: Optional[int] = ..., presence_penalty: float = ..., frequency_penalty: float = ..., temperature: float = ..., top_p: float = ..., top_k: int = ..., use_beam_search: bool = ..., stop: Union[None, str, List[str]] = ..., ignore_eos: bool = ..., max_tokens: int = ..., logprobs: Optional[int] = ...) -> None: ...
