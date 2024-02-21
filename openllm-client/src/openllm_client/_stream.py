from __future__ import annotations
import typing as t

import attr
import httpx
import orjson

if t.TYPE_CHECKING:
  from ._shim import AsyncClient, Client

Response = t.TypeVar('Response', bound=attr.AttrsInstance)


@attr.define(auto_attribs=True)
class Stream(t.Generic[Response]):
  _response_cls: t.Type[Response]
  _response: httpx.Response
  _client: Client
  _decoder: SSEDecoder = attr.field(factory=lambda: SSEDecoder())
  _iterator: t.Iterator[Response] = attr.field(init=False)

  def __init__(self, response_cls, response, client):
    self.__attrs_init__(response_cls, response, client)
    self._iterator = self._stream()

  def __next__(self):
    return self._iterator.__next__()

  def __iter__(self) -> t.Iterator[Response]:
    for item in self._iterator:
      yield item

  def _iter_events(self) -> t.Iterator[SSE]:
    yield from self._decoder.iter(self._response.iter_lines())

  def _stream(self) -> t.Iterator[Response]:
    for sse in self._iter_events():
      if sse.data.startswith('[DONE]'):
        break
      if sse.event is None:
        yield self._client._process_response_data(data=sse.model_dump(), response_cls=self._response_cls, raw_response=self._response)


@attr.define(auto_attribs=True)
class AsyncStream(t.Generic[Response]):
  _response_cls: t.Type[Response]
  _response: httpx.Response
  _client: AsyncClient
  _decoder: SSEDecoder = attr.field(factory=lambda: SSEDecoder())
  _iterator: t.Iterator[Response] = attr.field(init=False)

  def __init__(self, response_cls, response, client):
    self.__attrs_init__(response_cls, response, client)
    self._iterator = self._stream()

  async def __anext__(self):
    return await self._iterator.__anext__()

  async def __aiter__(self):
    async for item in self._iterator:
      yield item

  async def _iter_events(self):
    async for sse in self._decoder.aiter(self._response.aiter_lines()):
      yield sse

  async def _stream(self) -> t.AsyncGenerator[Response, None]:
    async for sse in self._iter_events():
      if sse.data.startswith('[DONE]'):
        break
      if sse.event is None:
        yield self._client._process_response_data(data=sse.model_dump(), response_cls=self._response_cls, raw_response=self._response)


@attr.define
class SSE:
  data: str = attr.field(default='')
  id: t.Optional[str] = attr.field(default=None)
  event: t.Optional[str] = attr.field(default=None)
  retry: t.Optional[int] = attr.field(default=None)

  def model_dump(self) -> t.Dict[str, t.Any]:
    try:
      return orjson.loads(self.data)
    except orjson.JSONDecodeError:
      raise


@attr.define(auto_attribs=True)
class SSEDecoder:
  _data: t.List[str] = attr.field(factory=list)
  _event: t.Optional[str] = None
  _retry: t.Optional[int] = None
  _last_event_id: t.Optional[str] = None

  def iter(self, iterator: t.Iterator[str]) -> t.Iterator[SSE]:
    for line in iterator:
      sse = self.decode(line.rstrip('\n'))
      if sse:
        yield sse

  async def aiter(self, iterator: t.AsyncIterator[str]) -> t.AsyncIterator[SSE]:
    async for line in iterator:
      sse = self.decode(line.rstrip('\n'))
      if sse:
        yield sse

  def decode(self, line: str) -> SSE | None:
    # NOTE: https://html.spec.whatwg.org/multipage/server-sent-events.html#event-stream-interpretation
    if not line:
      if all(not a for a in [self._event, self._data, self._retry, self._last_event_id]):
        return None
      sse = SSE(data='\n'.join(self._data), event=self._event, retry=self._retry, id=self._last_event_id)
      self._event, self._data, self._retry = None, [], None
      return sse
    if line.startswith(':'):
      return None
    field, _, value = line.partition(':')
    if value.startswith(' '):
      value = value[1:]
    if field == 'event':
      self._event = value
    elif field == 'data':
      self._data.append(value)
    elif field == 'id':
      if '\0' in value:
        pass
      else:
        self._last_event_id = value
    elif field == 'retry':
      try:
        self._retry = int(value)
      except (TypeError, ValueError):
        pass
    else:
      pass  # Ignore unknown fields
    return None
