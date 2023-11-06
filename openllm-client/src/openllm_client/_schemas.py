from __future__ import annotations
import typing as t

import attr
import cattr

if t.TYPE_CHECKING:
  from attr import AttrsInstance

class _Mixin:
  def model_dump_json(self) -> dict[str, t.Any]:
    if not attr.has(self.__class__): raise TypeError(f'Class {self.__class__} must be attr class')
    return cattr.unstructure(self)

  @classmethod
  def model_construct(cls, data: dict[str, t.Any]) -> type[AttrsInstance]:
    if not attr.has(cls): raise TypeError(f'Class {cls} must be attr class')
    return cattr.structure(data, cls)

# XXX: sync with openllm-core/src/openllm_core/_schemas.py
@attr.define
class Request(_Mixin):
  prompt: str
  llm_config: t.Dict[str, t.Any]
  return_type: t.Literal['token'] = 'token'
  stop: t.Optional[t.Union[str, t.List[str]]] = attr.field(default=None)
  adapter_name: t.Optional[str] = attr.field(default=None)

SampleLogprobs = t.List[t.Dict[int, float]]
PromptLogprobs = t.List[t.Optional[t.Dict[int, float]]]
FinishReason = t.Literal['length', 'stop']

@attr.define
class CompletionChunk:
  index: int
  text: str
  token_ids: t.List[int]
  cumulative_logprob: float
  logprobs: t.Optional[SampleLogprobs] = None
  finish_reason: t.Optional[FinishReason] = None

@attr.define
class Response(_Mixin):
  prompt: str
  finished: bool
  request_id: str
  outputs: t.List[CompletionChunk]
  prompt_token_ids: t.Optional[t.List[int]] = attr.field(default=None)
  prompt_logprobs: t.Optional[PromptLogprobs] = attr.field(default=None)

@attr.define
class StreamingResponse(_Mixin):
  request_id: str
  index: int
  text: str
  token_ids: int

  @classmethod
  def from_response_chunk(cls, response: Response) -> StreamingResponse:
    return cls(request_id=response.request_id, index=response.outputs[0].index, text=response.outputs[0].text, token_ids=response.outputs[0].token_ids[0])
