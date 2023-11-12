from __future__ import annotations
import typing as t

import attr
import cattr
import orjson

from ._utils import converter


if t.TYPE_CHECKING:
  from ._typing_compat import Self


@attr.define
class _SchemaMixin:
  def model_dump(self) -> t.Dict[str, t.Any]:
    return cattr.unstructure(self)

  def model_dump_json(self) -> str:
    return orjson.dumps(self.model_dump()).decode('utf-8')

  @classmethod
  def model_construct(cls, data: t.Dict[str, t.Any]) -> Self:
    return cattr.structure(data, cls)


# XXX: sync with openllm-core/src/openllm_core/_schemas.py
@attr.define
class MetadataOutput(_SchemaMixin):
  model_id: str
  timeout: int
  model_name: str
  backend: str
  configuration: t.Dict[str, t.Any]
  prompt_template: t.Optional[str]
  system_message: t.Optional[str]


def _structure_metadata(data: t.Dict[str, t.Any], cls: type[MetadataOutput]) -> MetadataOutput:
  try:
    configuration = orjson.loads(data['configuration'])
    generation_config = configuration.pop('generation_config')
    configuration = {**configuration, **generation_config}
  except orjson.JSONDecodeError as e:
    raise RuntimeError(f'Malformed metadata configuration (Server-side issue): {e}') from None
  try:
    return cls(
      model_id=data['model_id'],
      timeout=data['timeout'],
      model_name=data['model_name'],
      backend=data['backend'],
      configuration=configuration,
      prompt_template=data['prompt_template'],
      system_message=data['system_message'],
    )
  except Exception as e:
    raise RuntimeError(f'Malformed metadata (Server-side issue): {e}') from None


converter.register_structure_hook(MetadataOutput, _structure_metadata)


@attr.define
class Request(_SchemaMixin):
  prompt: str
  llm_config: t.Dict[str, t.Any]
  stop: t.Optional[t.Union[str, t.List[str]]] = attr.field(default=None)
  adapter_name: t.Optional[str] = attr.field(default=None)


SampleLogprobs = t.List[t.Dict[int, float]]
PromptLogprobs = t.List[t.Optional[t.Dict[int, float]]]
FinishReason = t.Literal['length', 'stop']


@attr.define
class CompletionChunk(_SchemaMixin):
  index: int
  text: str
  token_ids: t.List[int]
  cumulative_logprob: float
  logprobs: t.Optional[SampleLogprobs] = None
  finish_reason: t.Optional[FinishReason] = None


@attr.define
class Response(_SchemaMixin):
  prompt: str
  finished: bool
  request_id: str
  outputs: t.List[CompletionChunk]
  prompt_token_ids: t.Optional[t.List[int]] = attr.field(default=None)
  prompt_logprobs: t.Optional[PromptLogprobs] = attr.field(default=None)

  def model_dump_json(self) -> str:
    return orjson.dumps(self.model_dump(), option=orjson.OPT_NON_STR_KEYS).decode('utf-8')


@attr.define
class StreamingResponse(_SchemaMixin):
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
