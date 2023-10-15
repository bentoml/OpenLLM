from __future__ import annotations
import time
import typing as t

import attr

import openllm_core

@attr.define
class ErrorResponse:
  message: str
  type: str
  object: str = 'error'
  param: t.Optional[str] = None
  code: t.Optional[str] = None

@attr.define
class CompletionRequest:
  prompt: str
  model: str = attr.field(default=None)
  suffix: t.Optional[str] = attr.field(default=None)
  max_tokens: t.Optional[int] = attr.field(default=16)
  temperature: t.Optional[float] = attr.field(default=1.0)
  top_p: t.Optional[float] = attr.field(default=1.0)
  n: t.Optional[int] = attr.field(default=1)
  stream: t.Optional[bool] = attr.field(default=False)
  logprobs: t.Optional[int] = attr.field(default=None)
  echo: t.Optional[bool] = attr.field(default=False)
  stop: t.Optional[t.Union[str, t.List[str]]] = attr.field(default=None)
  presence_penalty: t.Optional[float] = attr.field(default=0.0)
  frequency_penalty: t.Optional[float] = attr.field(default=0.0)
  best_of: t.Optional[int] = attr.field(default=1)
  logit_bias: t.Optional[t.Dict[str, float]] = attr.field(default=None)
  user: t.Optional[str] = attr.field(default=None)

@attr.define
class ChatCompletionRequest:
  messages: t.List[t.Dict[str, str]]
  model: str = attr.field(default=None)
  functions: t.List[t.Dict[str, str]] = attr.field(default=attr.Factory(list))
  function_calls: t.List[t.Dict[str, str]] = attr.field(default=attr.Factory(list))
  temperature: t.Optional[float] = attr.field(default=None)
  top_p: t.Optional[float] = attr.field(default=None)
  n: t.Optional[int] = attr.field(default=None)
  stream: t.Optional[bool] = attr.field(default=False)
  stop: t.Optional[t.Union[str, t.List[str]]] = attr.field(default=None)
  max_tokens: t.Optional[int] = attr.field(default=None)
  presence_penalty: t.Optional[float] = attr.field(default=None)
  frequency_penalty: t.Optional[float] = attr.field(default=None)
  logit_bias: t.Optional[t.Dict[str, float]] = attr.field(default=None)
  user: t.Optional[str] = attr.field(default=None)

@attr.define
class LogProbs:
  text_offset: t.List[int] = attr.field(default=attr.Factory(list))
  token_logprobs: t.List[float] = attr.field(default=attr.Factory(list))
  tokens: t.List[str] = attr.field(default=attr.Factory(list))
  top_logprobs: t.List[t.Dict[str, t.Any]] = attr.field(default=attr.Factory(list))

@attr.define
class CompletionTextChoice:
  text: str
  index: int
  logprobs: LogProbs = attr.field(default=attr.Factory(lambda: LogProbs()))
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
  created: int = attr.field(default=attr.Factory(lambda: int(time.monotonic())))
  usage: Usage = attr.field(default=attr.Factory(lambda: Usage()))

@attr.define
class CompletionStreamResponse:
  choices: t.List[CompletionTextChoice]
  model: str
  object: str = 'text_completion'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('cmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.monotonic())))

LiteralRole = t.Literal['system', 'user', 'assistant']

class Message(t.TypedDict):
  role: LiteralRole
  content: str

@attr.define
class Delta:
  role: t.Optional[LiteralRole] = None
  content: t.Optional[str] = None

@attr.define
class ChatCompletionChoice:
  index: int
  message: Message
  finish_reason: str = attr.field(default=None)

@attr.define
class ChatCompletionResponseStreamChoice:
  index: int
  delta: Delta
  finish_reason: str = attr.field(default=None)

@attr.define
class ChatCompletionResponse:
  choices: t.List[ChatCompletionChoice]
  model: str
  object: str = 'chat.completion'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('chatcmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.monotonic())))
  usage: Usage = attr.field(default=attr.Factory(lambda: Usage()))

@attr.define
class ChatCompletionStreamResponse:
  choices: t.List[ChatCompletionResponseStreamChoice]
  model: str
  object: str = 'chat.completion.chunk'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('chatcmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.monotonic())))

@attr.define
class ModelCard:
  id: str
  object: str = 'model'
  created: int = attr.field(default=attr.Factory(lambda: int(time.monotonic())))
  owned_by: str = 'na'

@attr.define
class ModelList:
  object: str = 'list'
  data: t.List[ModelCard] = attr.field(factory=list)

async def get_conversation_prompt(request: ChatCompletionRequest) -> str:
  formatted = '\n'.join([f"{message['role']}: {message['content']}" for message in request.messages])
  return f'{formatted}\nassistant:'
