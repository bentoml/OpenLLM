from __future__ import annotations
import typing as t

import attr
import orjson

from openllm_core._schemas import CompletionChunk as CompletionChunk
from openllm_core._schemas import GenerationOutput as Response  # backward compatibility
from openllm_core._schemas import _SchemaMixin as _SchemaMixin

from ._utils import converter


__all__ = ['Response', 'CompletionChunk', 'Metadata', 'StreamingResponse']


@attr.define
class Metadata(_SchemaMixin):
  """NOTE: Metadata is a modified version of the original MetadataOutput from openllm-core.

  The configuration is now structured into a dictionary for easy of use."""

  model_id: str
  timeout: int
  model_name: str
  backend: str
  configuration: t.Dict[str, t.Any]
  prompt_template: t.Optional[str]
  system_message: t.Optional[str]


def _structure_metadata(data: t.Dict[str, t.Any], cls: type[Metadata]) -> Metadata:
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


converter.register_structure_hook(Metadata, _structure_metadata)


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
