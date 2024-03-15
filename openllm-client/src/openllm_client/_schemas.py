from __future__ import annotations
import types
import typing as t

import attr
import orjson

from openllm_core._schemas import (
  CompletionChunk as CompletionChunk,
  GenerationOutput as Response,  # backward compatibility
  _SchemaMixin as _SchemaMixin,
)

from ._utils import converter

if t.TYPE_CHECKING:
  from ._shim import AsyncClient, Client


__all__ = ['Response', 'CompletionChunk', 'Metadata', 'StreamingResponse', 'Helpers']


@attr.define
class Metadata(_SchemaMixin):
  """NOTE: Metadata is a modified version of the original MetadataOutput from openllm-core.

  The configuration is now structured into a dictionary for easy of use."""

  model_id: str
  timeout: int
  model_name: str
  backend: str
  configuration: t.Dict[str, t.Any]


def _structure_metadata(data: t.Dict[str, t.Any], cls: type[Metadata]) -> Metadata:
  try:
    configuration = orjson.loads(data['configuration'])
    generation_config = configuration.pop('generation_config')
    configuration = {**configuration, **generation_config}
  except orjson.JSONDecodeError as e:
    raise RuntimeError(f'Malformed metadata configuration (Server-side issue): {e}') from None
  try:
    return cls(
      model_id=data['model_id'], timeout=data['timeout'], model_name=data['model_name'], backend=data['backend'], configuration=configuration
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
      request_id=response.request_id, index=response.outputs[0].index, text=response.outputs[0].text, token_ids=response.outputs[0].token_ids[0]
    )


class MesssageParam(t.TypedDict):
  role: t.Literal['user', 'system', 'assistant']
  content: str


@attr.define(repr=False)
class Helpers:
  _client: t.Optional[Client] = None
  _async_client: t.Optional[AsyncClient] = None

  @property
  def client(self):
    if self._client is None:
      raise RuntimeError('Sync client should not be used within a async context.')
    return self._client

  @property
  def async_client(self):
    if self._async_client is None:
      raise RuntimeError('Async client should not be used within a sync context.')
    return self._async_client

  def messages(self, messages, add_generation_prompt=False):
    return self.client._post('/v1/helpers/messages', response_cls=str, json=dict(messages=messages, add_generation_prompt=add_generation_prompt))

  async def async_messages(self, messages, add_generation_prompt=False):
    return await self.async_client._post(
      '/v1/helpers/messages', response_cls=str, json=dict(messages=messages, add_generation_prompt=add_generation_prompt)
    )

  @classmethod
  def permute(cls, **attrs: t.Any) -> type[Helpers]:
    return types.new_class('Helpers', (cls,), {}, lambda ns: ns.update({'__module__': __name__, **attrs}))
