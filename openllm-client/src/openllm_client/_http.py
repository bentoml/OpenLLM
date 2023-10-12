from __future__ import annotations
import typing as t

from urllib.parse import urlparse

import attr
import httpx
import orjson

from ._schemas import Request
from ._schemas import Response
from ._schemas import StreamResponse

def _address_validator(_: t.Any, attr: attr.Attribute[t.Any], value: str) -> None:
  if not isinstance(value, str): raise TypeError(f'{attr.name} must be a string')
  if not urlparse(value).netloc: raise ValueError(f'{attr.name} must be a valid URL')

@attr.define
class HTTPClient:
  address: str = attr.field(validator=_address_validator, converter=lambda addr: addr if '://' in addr else 'http://' + addr)
  api_version: str = 'v1'
  timeout: int = 30
  client_args: t.Dict[str, t.Any] = attr.field(factory=dict)
  __metadata: dict[str, t.Any] = attr.field(default=None)
  __config: dict[str, t.Any] = attr.field(default=None)
  _inner: httpx.Client = attr.field(init=False, repr=False)

  def __attrs_post_init__(self) -> None:
    self._inner = httpx.Client(base_url=self.address, timeout=self.timeout, **self.client_args)

  def _metadata(self) -> dict[str, t.Any]:
    if self.__metadata is None: self.__metadata = self._inner.post(self._build_endpoint('metadata')).json()
    return self.__metadata

  def _config(self) -> dict[str, t.Any]:
    if self.__config is None:
      config = orjson.loads(self._metadata()['configuration'])
      generation_config = config.pop('generation_config')
      self.__config = {**config, **generation_config}
    return self.__config

  def health(self):
    return self._inner.get('/readyz')

  def _build_endpoint(self, endpoint: str):
    return '/' + f'{self.api_version}/{endpoint}'

  def query(self, prompt: str, **attrs: t.Any) -> Response:
    req = Request(prompt=self._metadata()['prompt_template'].format(system_message=self._metadata()['system_message'], instruction=prompt), llm_config={**self._config(), **attrs})
    r = self._inner.post(self._build_endpoint('generate'), json=req.json(), **self.client_args)
    payload = r.json()
    if r.status_code != 200: raise ValueError("Failed to get generation from '/v1/generate'. Check server logs for more details.")
    return Response(**payload)

  def generate(self, prompt: str, **attrs: t.Any) -> Response:
    return self.query(prompt, **attrs)

  def generate_stream(self, prompt: str, **attrs: t.Any) -> t.Iterator[StreamResponse]:
    req = Request(prompt=self._metadata()['prompt_template'].format(system_message=self._metadata()['system_message'], instruction=prompt), llm_config={**self._config(), **attrs})
    with self._inner.stream('POST', self._build_endpoint('generate_stream'), json=req.json(), **self.client_args) as r:
      for payload in r.iter_bytes():
        # Skip line
        payload = payload.decode('utf-8')
        yield StreamResponse(text=payload)
        # TODO: make it SSE correct for streaming
        # if payload == b"\n": continue
        # payload = payload.decode("utf-8")
        # if payload.startswith("data:"):
        #   json_payload = orjson.loads(payload.lstrip('data:').rstrip("\n"))
        #   print(json_payload)
        #   try: resp = StreamResponse(text=json_payload)
        #   except Exception as e: print(e)
        #   yield resp

  def __del__(self) -> None:
    self._inner.close()

@attr.define
class AsyncHTTPClient:
  address: str = attr.field(validator=_address_validator, converter=lambda addr: addr if '://' in addr else 'http://' + addr)
  api_version: str = 'v1'
  timeout: int = 30
  client_args: t.Dict[str, t.Any] = attr.field(factory=dict)
  __metadata: dict[str, t.Any] = attr.field(default=None)
  __config: dict[str, t.Any] = attr.field(default=None)
  _inner: httpx.AsyncClient = attr.field(init=False, repr=False)

  def __attrs_post_init__(self) -> None:
    self._inner = httpx.AsyncClient(base_url=self.address, timeout=self.timeout, **self.client_args)

  async def _metadata(self) -> dict[str, t.Any]:
    if self.__metadata is None: self.__metadata = (await self._inner.post(self._build_endpoint('metadata'))).json()
    return self.__metadata

  async def _config(self) -> dict[str, t.Any]:
    if self.__config is None:
      config = orjson.loads((await self._metadata())['configuration'])
      generation_config = config.pop('generation_config')
      self.__config = {**config, **generation_config}
    return self.__config

  async def health(self):
    return await self._inner.get('/readyz')

  def _build_endpoint(self, endpoint: str):
    return '/' + f'{self.api_version}/{endpoint}'

  async def query(self, prompt: str, **attrs: t.Any) -> Response:
    _meta, _config = await self._metadata(), await self._config()
    client = httpx.AsyncClient(base_url=self.address, timeout=self.timeout, **self.client_args)
    req = Request(prompt=_meta['prompt_template'].format(system_message=_meta['system_message'], instruction=prompt), llm_config={**_config, **attrs})
    r = await client.post(self._build_endpoint('generate'), json=req.json(), **self.client_args)
    payload = r.json()
    if r.status_code != 200: raise ValueError("Failed to get generation from '/v1/generate'. Check server logs for more details.")
    return Response(**payload)

  async def generate(self, prompt: str, **attrs: t.Any) -> Response:
    return await self.query(prompt, **attrs)

  async def generate_stream(self, prompt: str, **attrs: t.Any) -> t.AsyncGenerator[StreamResponse, t.Any]:
    _meta, _config = await self._metadata(), await self._config()
    client = httpx.AsyncClient(base_url=self.address, timeout=self.timeout, **self.client_args)
    req = Request(prompt=_meta['prompt_template'].format(system_message=_meta['system_message'], instruction=prompt), llm_config={**_config, **attrs})
    async with client.stream('POST', self._build_endpoint('generate_stream'), json=req.json(), **self.client_args) as r:
      async for payload in r.aiter_bytes():
        # Skip line
        payload = payload.decode('utf-8')
        yield StreamResponse(text=payload)
        # TODO: make it SSE correct for streaming
        # if payload == b"\n": continue
        # payload = payload.decode("utf-8")
        # if payload.startswith("data:"):
        #   json_payload = orjson.loads(payload.lstrip('data:').rstrip("\n"))
        #   print(json_payload)
        #   try: resp = StreamResponse(text=json_payload)
        #   except Exception as e: print(e)
        #   yield resp
