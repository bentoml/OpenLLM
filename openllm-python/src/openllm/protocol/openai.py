from __future__ import annotations
import typing as t
import time

import attr

import openllm_core

@attr.define
class LogProbs:
    text_offset: t.List[int] = attr.field(default=attr.Factory(list))
    token_logprobs: t.List[float] = attr.field(default=attr.Factory(list))
    tokens: t.List[str] = attr.field(default=attr.Factory(list))
    top_logprobs: t.List[dict] = attr.field(default=attr.Factory(list))

@attr.define
class CompletionTextChoice:
    text: str = attr.field(default='')
    index: int = attr.field(default=0)
    logprobs: LogProbs = attr.field(default=None)
    finish_reason: str = attr.field(default=None)

@attr.define
class Usage:
    prompt_tokens: int = attr.field(default=0)
    completion_tokens: int = attr.field(default=0)
    total_tokens: int = attr.field(default=0)

@attr.define
class CompletionResponse:
    id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('cmpl')))
    object: str = attr.field(default='text_completion')
    created: int = attr.field(default=attr.Factory(lambda: int(time.time())))
    model: str = attr.field(default=None)
    choices: list = attr.field(default=attr.Factory([CompletionTextChoice()]))
    usage: Usage = attr.field(default=attr.Factory(lambda: Usage()))

@attr.define
class CompletionResponseStream:
    id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('cmpl')))
    object: str = attr.field(default='text_completion')
    created: int = attr.field(default=attr.Factory(lambda: int(time.time())))
    choices: list = attr.field(default=attr.Factory([CompletionTextChoice()]))
    model: str = attr.field(default=None)

@attr.define
class Message:
    role: str = attr.field(default='')  # 'system', 'user', or 'assistant'
    content: str = attr.field(default='')

@attr.define
class Delta:
    role: t.Optional[str] = attr.field(default=None) # 'system', 'user', or 'assistant'
    content: t.Optional[str] = attr.field(default=None)

@attr.define
class ChatCompletionChoice:
    index: int = attr.field(default=0)
    message: Message = attr.field(default=attr.Factory(lambda: Message()))
    finish_reason: str = attr.field(default=None)

@attr.define
class ChatCompletionStreamChoice:
    index: int = attr.field(default=0)
    delta: Message = attr.field(default=attr.Factory(lambda: Delta()))
    finish_reason: str = attr.field(default=None)

@attr.define
class ChatCompletionResponse:
    id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('chatcmpl')))
    object: str = attr.field(default='chat.completion')
    created: int = attr.field(default=attr.Factory(lambda: int(time.time())))
    model: str = attr.field(default=None)
    choices: list = attr.field(default=attr.Factory([ChatCompletionChoice()]))
    usage: Usage = attr.field(default=attr.Factory(lambda: Usage()))

@attr.define
class ChatCompletionResponseStream:
    id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('chatcmpl')))
    object: str = attr.field(default='chat.completion.chunk')
    created: int = attr.field(default=attr.Factory(lambda: int(time.time())))
    model: str = attr.field(default=None)
    choices: list = attr.field(default=attr.Factory([ChatCompletionStreamChoice()]))