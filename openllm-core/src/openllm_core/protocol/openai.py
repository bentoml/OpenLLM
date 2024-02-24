from __future__ import annotations
import time, pydantic, typing as t
from .._schemas import FinishReason
from ..utils import gen_random_uuid


class ErrorResponse(pydantic.BaseModel):
  message: str
  type: str
  object: str = 'error'
  param: t.Optional[str] = None
  code: t.Optional[str] = None


class CompletionRequest(pydantic.BaseModel):
  prompt: str
  model: str = pydantic.Field(default=None)
  suffix: t.Optional[str] = pydantic.Field(default=None)
  max_tokens: t.Optional[int] = pydantic.Field(default=16)
  temperature: t.Optional[float] = pydantic.Field(default=1.0)
  top_p: t.Optional[float] = pydantic.Field(default=1.0)
  n: t.Optional[int] = pydantic.Field(default=1)
  stream: t.Optional[bool] = pydantic.Field(default=False)
  logprobs: t.Optional[int] = pydantic.Field(default=None)
  echo: t.Optional[bool] = pydantic.Field(default=False)
  stop: t.Optional[t.Union[str, t.List[str]]] = pydantic.Field(default=None)
  presence_penalty: t.Optional[float] = pydantic.Field(default=0.0)
  frequency_penalty: t.Optional[float] = pydantic.Field(default=0.0)
  logit_bias: t.Optional[t.Dict[str, float]] = pydantic.Field(default=None)
  user: t.Optional[str] = pydantic.Field(default=None)
  # supported by vLLM and us
  top_k: t.Optional[int] = pydantic.Field(default=None)
  best_of: t.Optional[int] = pydantic.Field(default=1)


class ChatCompletionRequest(pydantic.BaseModel):
  messages: t.List[t.Dict[str, str]]
  model: str = pydantic.Field(default=None)
  functions: t.List[t.Dict[str, str]] = pydantic.Field(default_factory=list)
  function_calls: t.List[t.Dict[str, str]] = pydantic.Field(default_factory=list)
  temperature: t.Optional[float] = pydantic.Field(default=None)
  top_p: t.Optional[float] = pydantic.Field(default=None)
  n: t.Optional[int] = pydantic.Field(default=None)
  stream: t.Optional[bool] = pydantic.Field(default=False)
  stop: t.Optional[t.Union[str, t.List[str]]] = pydantic.Field(default=None)
  max_tokens: t.Optional[int] = pydantic.Field(default=None)
  presence_penalty: t.Optional[float] = pydantic.Field(default=None)
  frequency_penalty: t.Optional[float] = pydantic.Field(default=None)
  echo: t.Optional[bool] = pydantic.Field(default=False)
  logit_bias: t.Optional[t.Dict[str, float]] = pydantic.Field(default=None)
  user: t.Optional[str] = pydantic.Field(default=None)
  # supported by vLLM and us
  top_k: t.Optional[int] = pydantic.Field(default=None)
  best_of: t.Optional[int] = pydantic.Field(default=1)
  # Additional features to support chat_template
  chat_template: str = pydantic.Field(default=None)
  add_generation_prompt: bool = pydantic.Field(default=True)


class LogProbs(pydantic.BaseModel):
  text_offset: t.List[int] = pydantic.Field(default_factory=list)
  token_logprobs: t.List[float] = pydantic.Field(default_factory=list)
  tokens: t.List[str] = pydantic.Field(default_factory=list)
  top_logprobs: t.List[t.Dict[str, t.Any]] = pydantic.Field(default_factory=list)


class UsageInfo(pydantic.BaseModel):
  prompt_tokens: int = pydantic.Field(default=0)
  completion_tokens: int = pydantic.Field(default=0)
  total_tokens: int = pydantic.Field(default=0)


class CompletionResponseChoice(pydantic.BaseModel):
  index: int
  text: str
  logprobs: t.Optional[LogProbs] = None
  finish_reason: t.Optional[FinishReason] = None


class CompletionResponseStreamChoice(pydantic.BaseModel):
  index: int
  text: str
  logprobs: t.Optional[LogProbs] = None
  finish_reason: t.Optional[FinishReason] = None


class CompletionStreamResponse(pydantic.BaseModel):
  model: str
  choices: t.List[CompletionResponseStreamChoice]
  object: str = pydantic.Field(default='text_completion')
  id: str = pydantic.Field(default_factory=lambda: gen_random_uuid('cmpl'))
  created: int = pydantic.Field(default_factory=lambda: int(time.monotonic()))
  usage: t.Optional[UsageInfo] = pydantic.Field(default=None)


class CompletionResponse(pydantic.BaseModel):
  choices: t.List[CompletionResponseChoice]
  model: str
  usage: UsageInfo
  object: str = pydantic.Field(default='text_completion')
  id: str = pydantic.Field(default_factory=lambda: gen_random_uuid('cmpl'))
  created: int = pydantic.Field(default_factory=lambda: int(time.monotonic()))


LiteralRole = t.Literal['system', 'user', 'assistant']


class Delta(pydantic.BaseModel):
  role: t.Optional[LiteralRole] = None
  content: t.Optional[str] = None


class ChatMessage(pydantic.BaseModel):
  role: LiteralRole
  content: str


class ChatCompletionResponseStreamChoice(pydantic.BaseModel):
  index: int
  delta: Delta
  finish_reason: t.Optional[FinishReason] = None


class ChatCompletionResponseChoice(pydantic.BaseModel):
  index: int
  message: ChatMessage
  finish_reason: t.Optional[FinishReason] = None


class ChatCompletionResponse(pydantic.BaseModel):
  choices: t.List[ChatCompletionResponseChoice]
  model: str
  object: str = pydantic.Field(default='chat.completion')
  id: str = pydantic.Field(default_factory=lambda: gen_random_uuid('chatcmpl'))
  created: int = pydantic.Field(default_factory=lambda: int(time.monotonic()))
  usage: UsageInfo = pydantic.Field(default_factory=lambda: UsageInfo())


class ChatCompletionStreamResponse(pydantic.BaseModel):
  choices: t.List[ChatCompletionResponseStreamChoice]
  model: str
  object: str = pydantic.Field(default='chat.completion.chunk')
  id: str = pydantic.Field(default_factory=lambda: gen_random_uuid('chatcmpl'))
  created: int = pydantic.Field(default_factory=lambda: int(time.monotonic()))
  usage: t.Optional[UsageInfo] = pydantic.Field(default=None)


class ModelCard(pydantic.BaseModel):
  id: str
  object: str = 'model'
  created: int = pydantic.Field(default_factory=lambda: int(time.monotonic()))
  owned_by: str = 'na'


class ModelList(pydantic.BaseModel):
  object: str = 'list'
  data: t.List[ModelCard] = pydantic.Field(default_factory=list)
