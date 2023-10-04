from __future__ import annotations
import time
import typing as t

import attr

import openllm_core

@attr.define
class LogProbs:
  text_offset: t.List[int]
  token_logprobs: t.List[float]
  tokens: t.List[str]
  top_logprobs: t.List[t.Dict[str, t.Any]]

@attr.define
class CompletionTextChoice:
  text: str
  index: int
  logprobs: LogProbs = attr.field(default=None)
  finish_reason: str = attr.field(default=None)

@attr.define
class Usage:
  prompt_tokens: int = attr.field(default=0)
  completion_tokens: int = attr.field(default=0)
  total_tokens: int = attr.field(default=0)

@attr.define
class CompletionResponse:
  choices: t.List[CompletionTextChoice]
  model: str
  object: str = 'text_completion'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('cmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.time())))
  usage: Usage = attr.field(default=attr.Factory(lambda: Usage()))

@attr.define
class CompletionResponseStream:
  choices: t.List[CompletionTextChoice]
  model: str
  object: str = 'text_completion'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('cmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.time())))

LiteralRole = t.Literal['system', 'user', 'assistant']

class Message(t.TypedDict):
  role: LiteralRole
  content: str

@attr.define
class Delta:
  role: LiteralRole
  content: str

@attr.define
class ChatCompletionChoice:
  index: int
  message: Message
  finish_reason: str = attr.field(default=None)

@attr.define
class ChatCompletionStreamChoice:
  index: int
  delta: Message
  finish_reason: str = attr.field(default=None)

@attr.define
class ChatCompletionResponse:
  choices: t.List[ChatCompletionChoice]
  model: str
  object: str = 'chat.completion'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('chatcmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.time())))
  usage: Usage = attr.field(default=attr.Factory(lambda: Usage()))

@attr.define
class ChatCompletionResponseStream:
  choices: t.List[ChatCompletionStreamChoice]
  model: str
  object: str = 'chat.completion.chunk'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('chatcmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.time())))

def openai_messages_to_openllm_prompt(messages: list[Message]) -> str:
  # TODO: Improve the prompt
  formatted = '\n'.join([f"{message['role']}: {message['content']}" for message in messages])
  return f"Complete the assistant's response. Use system info if provided.\n{formatted}\nassistant:"
