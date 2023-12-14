from __future__ import annotations
import time
import typing as t

import attr

import openllm_core
from openllm_core._schemas import FinishReason
from openllm_core.utils import converter


@attr.define
class ErrorResponse:
  message: str
  type: str
  object: str = 'error'
  param: t.Optional[str] = None
  code: t.Optional[str] = None


def _stop_converter(data: t.Union[str, t.List[str]]) -> t.List[str]:
  if not data:
    return None
  return [data] if isinstance(data, str) else data


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
  stop: t.Optional[t.Union[str, t.List[str]]] = attr.field(default=None, converter=_stop_converter)
  presence_penalty: t.Optional[float] = attr.field(default=0.0)
  frequency_penalty: t.Optional[float] = attr.field(default=0.0)
  logit_bias: t.Optional[t.Dict[str, float]] = attr.field(default=None)
  user: t.Optional[str] = attr.field(default=None)
  # supported by vLLM and us
  top_k: t.Optional[int] = attr.field(default=None)
  best_of: t.Optional[int] = attr.field(default=1)


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
  stop: t.Optional[t.Union[str, t.List[str]]] = attr.field(default=None, converter=_stop_converter)
  max_tokens: t.Optional[int] = attr.field(default=None)
  presence_penalty: t.Optional[float] = attr.field(default=None)
  frequency_penalty: t.Optional[float] = attr.field(default=None)
  echo: t.Optional[bool] = attr.field(default=False)
  logit_bias: t.Optional[t.Dict[str, float]] = attr.field(default=None)
  user: t.Optional[str] = attr.field(default=None)
  # supported by vLLM and us
  top_k: t.Optional[int] = attr.field(default=None)
  best_of: t.Optional[int] = attr.field(default=1)
  # Additional features to support chat_template
  chat_template: str = attr.field(default=None)
  add_generation_prompt: bool = attr.field(default=True)


@attr.define
class LogProbs:
  text_offset: t.List[int] = attr.field(default=attr.Factory(list))
  token_logprobs: t.List[float] = attr.field(default=attr.Factory(list))
  tokens: t.List[str] = attr.field(default=attr.Factory(list))
  top_logprobs: t.List[t.Dict[str, t.Any]] = attr.field(default=attr.Factory(list))


@attr.define
class UsageInfo:
  prompt_tokens: int = attr.field(default=0)
  completion_tokens: int = attr.field(default=0)
  total_tokens: int = attr.field(default=0)


@attr.define
class CompletionResponseChoice:
  index: int
  text: str
  logprobs: t.Optional[LogProbs] = None
  finish_reason: t.Optional[FinishReason] = None


@attr.define
class CompletionResponseStreamChoice:
  index: int
  text: str
  logprobs: t.Optional[LogProbs] = None
  finish_reason: t.Optional[FinishReason] = None


@attr.define
class CompletionStreamResponse:
  model: str
  choices: t.List[CompletionResponseStreamChoice]
  object: str = 'text_completion'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('cmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.monotonic())))
  usage: t.Optional[UsageInfo] = attr.field(default=None)


@attr.define
class CompletionResponse:
  choices: t.List[CompletionResponseChoice]
  model: str
  usage: UsageInfo
  object: str = 'text_completion'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('cmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.monotonic())))


LiteralRole = t.Literal['system', 'user', 'assistant']


@attr.define
class Delta:
  role: t.Optional[LiteralRole] = None
  content: t.Optional[str] = None


@attr.define
class ChatMessage:
  role: LiteralRole
  content: str


converter.register_unstructure_hook(ChatMessage, lambda msg: {'role': msg.role, 'content': msg.content})


@attr.define
class ChatCompletionResponseStreamChoice:
  index: int
  delta: Delta
  finish_reason: t.Optional[FinishReason] = None


@attr.define
class ChatCompletionResponseChoice:
  index: int
  message: ChatMessage
  finish_reason: t.Optional[FinishReason] = None


@attr.define
class ChatCompletionResponse:
  choices: t.List[ChatCompletionResponseChoice]
  model: str
  object: str = 'chat.completion'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('chatcmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.monotonic())))
  usage: UsageInfo = attr.field(default=attr.Factory(lambda: UsageInfo()))


@attr.define
class ChatCompletionStreamResponse:
  choices: t.List[ChatCompletionResponseStreamChoice]
  model: str
  object: str = 'chat.completion.chunk'
  id: str = attr.field(default=attr.Factory(lambda: openllm_core.utils.gen_random_uuid('chatcmpl')))
  created: int = attr.field(default=attr.Factory(lambda: int(time.monotonic())))
  usage: t.Optional[UsageInfo] = attr.field(default=None)


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
