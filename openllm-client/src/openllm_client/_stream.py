from __future__ import annotations

import pydantic, httpx, orjson, typing as t

if t.TYPE_CHECKING:
  from ._shim import AsyncClient, Client

Response = t.TypeVar('Response', bound=pydantic.BaseModel)


class Stream(pydantic.BaseModel, t.Generic[Response]):
  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
  response_cls: t.Type[Response]
  response: pydantic.SkipValidation[httpx.Response]
  client: Client
  _decoder: SSEDecoder = pydantic.PrivateAttr(default_factory=lambda: SSEDecoder())
  _iterator: t.Iterator[Response] = pydantic.PrivateAttr()

  def __init__(self, **data):
    super().__init__(**data)
    self._iterator = self._stream()

  def __next__(self):
    return self._iterator.__next__()

  def __iter__(self) -> t.Iterator[Response]:
    for item in self._iterator:
      yield item

  def _iter_events(self) -> t.Iterator[SSE]:
    yield from self._decoder.iter(self.response.iter_lines())

  def _stream(self) -> t.Iterator[Response]:
    for sse in self._iter_events():
      if sse.data.startswith('[DONE]'):
        break
      if sse.event is None:
        yield self.response_cls(**orjson.loads(sse.data))


class AsyncStream(pydantic.BaseModel, t.Generic[Response]):
  model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
  response_cls: t.Type[Response]
  response: pydantic.SkipValidation[httpx.Response]
  client: AsyncClient
  _decoder: SSEDecoder = pydantic.PrivateAttr(default_factory=lambda: SSEDecoder())
  _iterator: t.Iterator[Response] = pydantic.PrivateAttr()

  def __init__(self, **data):
    super().__init__(**data)
    self._iterator = self._stream()

  async def __anext__(self):
    return await self._iterator.__anext__()

  async def __aiter__(self):
    async for item in self._iterator:
      yield item

  async def _iter_events(self):
    async for sse in self._decoder.aiter(self.response.aiter_lines()):
      yield sse

  async def _stream(self) -> t.AsyncGenerator[Response, None]:
    async for sse in self._iter_events():
      if sse.data.startswith('[DONE]'):
        break
      if sse.event is None:
        yield self.response_cls(**orjson.loads(sse.data))


class SSE(pydantic.BaseModel):
  data: str = pydantic.Field(default='')
  id: t.Optional[str] = pydantic.Field(default=None)
  event: t.Optional[str] = pydantic.Field(default=None)
  retry: t.Optional[int] = pydantic.Field(default=None)


class SSEDecoder(pydantic.BaseModel):
  _data: t.List[str] = pydantic.PrivateAttr(default_factory=list)
  event: t.Optional[str] = None
  retry: t.Optional[int] = None
  last_event_id: t.Optional[str] = None

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
      if all(not a for a in [self.event, self._data, self.retry, self.last_event_id]):
        return None
      sse = SSE(data='\n'.join(self._data), event=self.event, retry=self.retry, id=self.last_event_id)
      self.event, self._data, self.retry = None, [], None
      return sse
    if line.startswith(':'):
      return None
    field, _, value = line.partition(':')
    if value.startswith(' '):
      value = value[1:]
    if field == 'event':
      self.event = value
    elif field == 'data':
      self._data.append(value)
    elif field == 'id':
      if '\0' in value:
        pass
      else:
        self.last_event_id = value
    elif field == 'retry':
      try:
        self.retry = int(value)
      except (TypeError, ValueError):
        pass
    else:
      pass  # Ignore unknown fields
    return None
