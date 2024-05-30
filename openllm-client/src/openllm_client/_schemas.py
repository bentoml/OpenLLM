from __future__ import annotations

import types, pydantic, typing as t

from openllm_core._schemas import (
  CompletionChunk as CompletionChunk,
  MetadataOutput as Metadata,
  GenerationOutput as Response,
)
from ._typing_compat import TypedDict

if t.TYPE_CHECKING:
  from ._shim import AsyncClient, Client


__all__ = ['CompletionChunk', 'Helpers', 'Metadata', 'Response', 'StreamingResponse']


class StreamingResponse(pydantic.BaseModel):
  request_id: str
  index: int
  text: str
  token_ids: int

  @classmethod
  def from_response_chunk(cls, response: Response, index: int = 0) -> StreamingResponse:
    return cls(
      request_id=response.request_id,
      index=response.outputs[index].index,
      text=response.outputs[index].text,
      token_ids=response.outputs[index].token_ids[0],
    )


class MesssageParam(TypedDict):
  role: t.Literal['user', 'system', 'assistant']
  content: str


class Helpers(pydantic.BaseModel):
  client: t.Optional[Client] = None
  async_client: t.Optional[AsyncClient] = None

  def messages(self, messages, add_generation_prompt=False):
    if self.client is None:
      raise RuntimeError('client is not initialised correctly.')
    return self.client._post(
      '/v1/helpers/messages',
      response_cls=str,
      json=dict(messages=messages, add_generation_prompt=add_generation_prompt),
    )

  async def async_messages(self, messages, add_generation_prompt=False):
    if self.async_client is None:
      raise RuntimeError('async_client is not initialised correctly.')
    return await self.async_client._post(
      '/v1/helpers/messages',
      response_cls=str,
      json=dict(messages=messages, add_generation_prompt=add_generation_prompt),
    )

  @classmethod
  def permute(cls, **attrs: t.Any) -> type[Helpers]:
    return types.new_class('Helpers', (cls,), {}, lambda ns: ns.update({'__module__': __name__, **attrs}))
