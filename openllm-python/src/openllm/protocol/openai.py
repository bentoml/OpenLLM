import attr
import openllm_core
import time

@attr.define
class LogProbs:
    text_offset: list[int] = attr.ib(default=attr.Factory(list))
    token_logprobs: list[float] = attr.ib(default=attr.Factory(list))
    tokens: list[str] = attr.ib(default=attr.Factory(list))
    top_logprobs: list[dict] = attr.ib(default=attr.Factory(list))

@attr.define
class CompletionTextChoice:
    text: str = attr.ib(default='')
    index: int = attr.ib(default=0)
    logprobs: LogProbs = attr.ib(default=None)
    finish_reason: str = attr.ib(default=None)

@attr.define
class Usage:
    prompt_tokens: int = attr.ib(default=0)
    completion_tokens: int = attr.ib(default=0)
    total_tokens: int = attr.ib(default=0)

@attr.define
class CompletionResponse:
    id: str = attr.ib(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('cmpl')))
    object: str = attr.ib(default='text_completion')
    created: int = attr.ib(default=attr.Factory(lambda: int(time.time())))
    model: str = attr.ib(default=None)
    choices: list = attr.ib(default=attr.Factory([CompletionTextChoice()]))
    usage: Usage = attr.ib(default=attr.Factory(lambda: Usage()))
    
@attr.define
class CompletionResponseStream:
    id: str = attr.ib(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('cmpl')))
    object: str = attr.ib(default='text_completion')
    created: int = attr.ib(default=attr.Factory(lambda: int(time.time())))
    choices: list = attr.ib(default=attr.Factory([CompletionTextChoice()]))
    model: str = attr.ib(default=None)




