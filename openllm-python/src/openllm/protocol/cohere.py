from __future__ import annotations
import typing as t

import attr

from openllm_core.utils import converter


@attr.define
class CohereErrorResponse:
  text: str


converter.register_unstructure_hook(CohereErrorResponse, lambda obj: obj.text)


@attr.define
class CohereGenerateRequest:
  prompt: str
  prompt_vars: t.Optional[t.Dict[str, t.Any]] = None
  model: t.Optional[str] = None
  preset: t.Optional[str] = None
  num_generations: t.Optional[int] = None
  max_tokens: t.Optional[int] = None
  temperature: t.Optional[float] = None
  k: t.Optional[int] = None
  p: t.Optional[float] = None
  frequency_penalty: t.Optional[float] = None
  presence_penalty: t.Optional[float] = None
  end_sequences: t.Optional[t.List[str]] = None
  stop_sequences: t.Optional[t.List[str]] = None
  return_likelihoods: t.Optional[t.Literal['GENERATION', 'ALL', 'NONE']] = None
  truncate: t.Optional[str] = None
  logit_bias: t.Optional[t.Dict[int, float]] = None
  stream: bool = False


@attr.define
class CohereChatRequest:
  message: str
  conversation_id: t.Optional[str] = ''
  model: t.Optional[str] = None
  return_chat_history: t.Optional[bool] = False
  return_prompt: t.Optional[bool] = False
  return_preamble: t.Optional[bool] = False
  chat_history: t.Optional[t.List[t.Dict[str, str]]] = None
  preamble_override: t.Optional[str] = None
  user_name: t.Optional[str] = None
  temperature: t.Optional[float] = 0.8
  max_tokens: t.Optional[int] = None
  stream: t.Optional[bool] = False
  p: t.Optional[float] = None
  k: t.Optional[float] = None
  logit_bias: t.Optional[t.Dict[int, float]] = None
  search_queries_only: t.Optional[bool] = None
  documents: t.Optional[t.List[t.Dict[str, t.Any]]] = None
  citation_quality: t.Optional[str] = None
  prompt_truncation: t.Optional[str] = None
  connectors: t.Optional[t.List[t.Dict[str, t.Any]]] = None


@attr.define
class TokenLikelihood:  # pretty sure this is similar to token_logprobs
  token: str
  likelihood: float


@attr.define
class Generation:
  id: str
  text: str
  prompt: str
  likelihood: t.Optional[float] = None
  token_likelihoods: t.List[TokenLikelihood] = attr.field(factory=list)
  finish_reason: t.Optional[str] = None


@attr.define
class Generations:
  id: str
  generations: t.List[Generation]
  meta: t.Optional[t.Dict[str, t.Any]] = None


@attr.define
class StreamingText:
  index: int
  text: str
  is_finished: bool


@attr.define
class StreamingGenerations:
  id: str
  generations: Generations
  texts: t.List[str]
  meta: t.Optional[t.Dict[str, t.Any]] = None
