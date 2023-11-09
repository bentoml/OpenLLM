from __future__ import annotations
import typing as t

import attr
import cattr


# XXX: sync with openllm-core/src/openllm_core/_schemas.py
@attr.define
class Request:
  prompt: str
  llm_config: t.Dict[str, t.Any]
  stop: t.Optional[t.Union[str, t.List[str]]] = attr.field(default=None)
  adapter_name: t.Optional[str] = attr.field(default=None)

  def model_dump_json(self) -> t.Dict[str, t.Any]:
    return cattr.unstructure(self)

  @classmethod
  def model_construct(cls, data: t.Dict[str, t.Any]) -> Request:
    return cattr.structure(data, cls)


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
class Response:
  prompt: str
  finished: bool
  request_id: str
  outputs: t.List[CompletionChunk]
  prompt_token_ids: t.Optional[t.List[int]] = attr.field(default=None)
  prompt_logprobs: t.Optional[PromptLogprobs] = attr.field(default=None)

  def model_dump_json(self) -> t.Dict[str, t.Any]:
    return cattr.unstructure(self)

  @classmethod
  def model_construct(cls, data: t.Dict[str, t.Any]) -> Response:
    return cattr.structure(data, cls)


@attr.define
class StreamingResponse:
  request_id: str
  index: int
  text: str
  token_ids: int

  @classmethod
  def from_response_chunk(cls, response: Response) -> StreamingResponse:
    return cls(
      request_id=response.request_id,
      index=response.outputs[0].index,
      text=response.outputs[0].text,
      token_ids=response.outputs[0].token_ids[0],
    )

  def model_dump_json(self) -> t.Dict[str, t.Any]:
    return cattr.unstructure(self)

  @classmethod
  def model_construct(cls, data: t.Dict[str, t.Any]) -> StreamingResponse:
    return cattr.structure(data, cls)
