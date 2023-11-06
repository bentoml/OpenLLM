from __future__ import annotations
import asyncio
import enum
import logging
import os
import time
import typing as t
import urllib.error

from urllib.parse import urlparse

import attr
import httpx
import orjson

from ._schemas import Request
from ._schemas import Response
from ._schemas import StreamingResponse

logger = logging.getLogger(__name__)

def _address_validator(_: t.Any, attr: attr.Attribute[t.Any], value: str) -> None:
  if not isinstance(value, str): raise TypeError(f'{attr.name} must be a string')
  if not urlparse(value).netloc: raise ValueError(f'{attr.name} must be a valid URL')

def _address_converter(addr: str) -> str:
  return addr if '://' in addr else 'http://' + addr

class ServerState(enum.Enum):
  # CLOSED: The server is not yet ready or `wait_until_server_ready` has not been called/failed.
  CLOSED = 1
  # READY: The server is ready and `wait_until_server_ready` has been called.
  READY = 2

_object_setattr = object.__setattr__

@attr.define(init=False)
class HTTPClient:
  address: str = attr.field(validator=_address_validator, converter=_address_converter)
  client_args: t.Dict[str, t.Any] = attr.field()
  _inner: httpx.Client = attr.field(repr=False)

  _timeout: int = attr.field(default=30, repr=False)
  _api_version: str = attr.field(default='v1', repr=False)
  _state: ServerState = attr.field(default=ServerState.CLOSED, repr=False)

  __metadata: dict[str, t.Any] | None = attr.field(default=None, repr=False)
  __config: dict[str, t.Any] | None = attr.field(default=None, repr=False)

  @staticmethod
  def wait_until_server_ready(addr: str, timeout: float = 30, check_interval: int = 1, **client_args: t.Any) -> None:
    addr = _address_converter(addr)
    logger.debug('Wait for server @ %s to be ready', addr)
    start = time.monotonic()
    while time.monotonic() - start < timeout:
      try:
        with httpx.Client(base_url=addr, **client_args) as sess:
          status = sess.get('/readyz').status_code
        if status == 200: break
        else: time.sleep(check_interval)
      except (httpx.ConnectError, urllib.error.URLError, ConnectionError):
        logger.debug('Server is not ready yet, retrying in %d seconds...', check_interval)
        time.sleep(check_interval)
    # Try once more and raise for exception
    try:
      with httpx.Client(base_url=addr, **client_args) as sess:
        status = sess.get('/readyz').status_code
    except httpx.HTTPStatusError as err:
      logger.error('Failed to wait until server ready: %s', addr)
      logger.error(err)
      raise

  def __init__(self, address: str | None = None, timeout: int = 30, api_version: str = 'v1', **client_args: t.Any) -> None:
    if address is None:
      env = os.environ.get('OPENLLM_ENDPOINT')
      if env is None: raise ValueError('address must be provided')
      address = env
    self.__attrs_init__(address, client_args, httpx.Client(base_url=address, timeout=timeout, **client_args), timeout, api_version)  # type: ignore[attr-defined]

  def _metadata(self) -> dict[str, t.Any]:
    if self.__metadata is None: self.__metadata = self._inner.post(self._build_endpoint('metadata')).json()
    return self.__metadata

  def _config(self) -> dict[str, t.Any]:
    if self.__config is None:
      config = orjson.loads(self._metadata()['configuration'])
      generation_config = config.pop('generation_config')
      self.__config = {**config, **generation_config}
    return self.__config

  # yapf: disable
  def __del__(self)->None:self._inner.close()
  def _build_endpoint(self,endpoint: str)->str:return ('/' if not self._api_version.startswith('/') else '')+f'{self._api_version}/{endpoint}'
  @property
  def is_ready(self)->bool:return self._state==ServerState.READY
  def query(self, prompt: str, **attrs: t.Any)->Response: return self.generate(prompt,**attrs)
  # yapf: enable

  def health(self) -> None:
    try:
      self.wait_until_server_ready(self.address, timeout=self._timeout, **self.client_args)
      _object_setattr(self, '_state', ServerState.READY)
    except Exception as e:
      logger.error('Server is not healthy (Scroll up for traceback)\n%s', e)
      _object_setattr(self, '_state', ServerState.CLOSED)

  def generate(self, prompt: str, llm_config: dict[str, t.Any] | None = None, stop: str | list[str] | None = None, adapter_name: str | None = None, **attrs: t.Any) -> Response:
    if not self.is_ready:
      self.health()
      if not self.is_ready: raise RuntimeError('Server is not ready. Check server logs for more information.')
    timeout = attrs.pop('timeout', self._timeout)
    _meta, _config = self._metadata(), self._config()
    if llm_config is not None: llm_config = {**_config, **llm_config, **attrs}
    else: llm_config = {**_config, **attrs}
    if _meta['prompt_template'] is not None: prompt = _meta['prompt_template'].format(system_message=_meta['system_message'], instruction=prompt)

    req = Request(prompt=prompt, llm_config=llm_config, stop=stop, adapter_name=adapter_name)
    with httpx.Client(base_url=self.address, timeout=timeout, **self.client_args) as client:
      r = client.post(self._build_endpoint('generate'), json=req.model_dump_json(), **self.client_args)
    if r.status_code != 200: raise ValueError("Failed to get generation from '/v1/generate'. Check server logs for more details.")
    return Response(**r.json())

  def generate_stream(self,
                      prompt: str,
                      llm_config: dict[str, t.Any] | None = None,
                      stop: str | list[str] | None = None,
                      adapter_name: str | None = None,
                      **attrs: t.Any) -> t.Iterator[StreamingResponse]:
    if not self.is_ready:
      self.health()
      if not self.is_ready: raise RuntimeError('Server is not ready. Check server logs for more information.')
    timeout = attrs.pop('timeout', self._timeout)
    _meta, _config = self._metadata(), self._config()
    if llm_config is not None: llm_config = {**_config, **llm_config, **attrs}
    else: llm_config = {**_config, **attrs}
    if _meta['prompt_template'] is not None: prompt = _meta['prompt_template'].format(system_message=_meta['system_message'], instruction=prompt)

    req = Request(prompt=prompt, llm_config=llm_config, stop=stop, adapter_name=adapter_name)
    with httpx.Client(base_url=self.address, timeout=timeout, **self.client_args) as client:
      with client.stream('POST', self._build_endpoint('generate_stream'), json=req.model_dump_json(), **self.client_args) as r:
        for payload in r.iter_bytes():
          if payload == b'data: [DONE]\n\n': break
          # Skip line
          if payload == b'\n': continue
          if payload.startswith(b'data: '):
            try:
              proc = payload.decode('utf-8').lstrip('data: ').rstrip('\n')
              data = orjson.loads(proc)
            except Exception:
              pass  # FIXME: Handle this
            yield StreamingResponse.from_response_chunk(Response.model_construct(data))

@attr.define(init=False)
class AsyncHTTPClient:
  address: str = attr.field(validator=_address_validator, converter=_address_converter)
  client_args: t.Dict[str, t.Any] = attr.field()
  _inner: httpx.AsyncClient = attr.field(repr=False)

  _timeout: int = attr.field(default=30, repr=False)
  _api_version: str = attr.field(default='v1', repr=False)
  _state: ServerState = attr.field(default=ServerState.CLOSED, repr=False)

  __metadata: dict[str, t.Any] | None = attr.field(default=None, repr=False)
  __config: dict[str, t.Any] | None = attr.field(default=None, repr=False)

  @staticmethod
  async def wait_until_server_ready(addr: str, timeout: float = 30, check_interval: int = 1, **client_args: t.Any) -> None:
    addr = _address_converter(addr)
    logger.debug('Wait for server @ %s to be ready', addr)
    start = time.monotonic()
    while time.monotonic() - start < timeout:
      try:
        async with httpx.AsyncClient(base_url=addr, **client_args) as sess:
          status = (await sess.get('/readyz')).status_code
        if status == 200: break
        else: await asyncio.sleep(check_interval)
      except (httpx.ConnectError, urllib.error.URLError, ConnectionError):
        logger.debug('Server is not ready yet, retrying in %d seconds...', check_interval)
        await asyncio.sleep(check_interval)
    # Try once more and raise for exception
    try:
      async with httpx.AsyncClient(base_url=addr, **client_args) as sess:
        status = (await sess.get('/readyz')).status_code
    except httpx.HTTPStatusError as err:
      logger.error('Failed to wait until server ready: %s', addr)
      logger.error(err)
      raise

  def __init__(self, address: str | None = None, timeout: int = 30, api_version: str = 'v1', **client_args: t.Any) -> None:
    if address is None:
      env = os.environ.get('OPENLLM_ENDPOINT')
      if env is None: raise ValueError('address must be provided')
      address = env
    self.__attrs_init__(address, client_args, httpx.AsyncClient(base_url=address, timeout=timeout, **client_args), timeout, api_version)

  async def _metadata(self) -> dict[str, t.Any]:
    if self.__metadata is None: self.__metadata = (await self._inner.post(self._build_endpoint('metadata'))).json()
    return self.__metadata

  async def _config(self) -> dict[str, t.Any]:
    if self.__config is None:
      config = orjson.loads((await self._metadata())['configuration'])
      generation_config = config.pop('generation_config')
      self.__config = {**config, **generation_config}
    return self.__config

  # yapf: disable
  def _build_endpoint(self,endpoint:str) -> str: return '/'+f'{self._api_version}/{endpoint}'
  @property
  def is_ready(self)->bool:return self._state==ServerState.READY
  async def query(self,prompt:str,**attrs: t.Any)->Response: return await self.generate(prompt,**attrs)
  # yapf: enable
  async def health(self) -> None:
    try:
      await self.wait_until_server_ready(self.address, timeout=self._timeout, **self.client_args)
      _object_setattr(self, '_state', ServerState.READY)
    except Exception as e:
      logger.error('Server is not healthy (Scroll up for traceback)\n%s', e)
      _object_setattr(self, '_state', ServerState.CLOSED)

  async def generate(self, prompt: str, llm_config: dict[str, t.Any] | None = None, stop: str | list[str] | None = None, adapter_name: str | None = None, **attrs: t.Any) -> Response:
    if not self.is_ready:
      await self.health()
      if not self.is_ready: raise RuntimeError('Server is not ready. Check server logs for more information.')
    timeout = attrs.pop('timeout', self._timeout)
    _meta, _config = await self._metadata(), await self._config()
    if llm_config is not None: llm_config = {**_config, **llm_config, **attrs}
    else: llm_config = {**_config, **attrs}
    if _meta['prompt_template'] is not None: prompt = _meta['prompt_template'].format(system_message=_meta['system_message'], instruction=prompt)

    req = Request(prompt=prompt, llm_config=llm_config, stop=stop, adapter_name=adapter_name)
    async with httpx.AsyncClient(base_url=self.address, timeout=timeout, **self.client_args) as client:
      r = await client.post(self._build_endpoint('generate'), json=req.model_dump_json(), **self.client_args)
    if r.status_code != 200: raise ValueError("Failed to get generation from '/v1/generate'. Check server logs for more details.")
    return Response(**r.json())

  async def generate_stream(self,
                            prompt: str,
                            llm_config: dict[str, t.Any] | None = None,
                            stop: str | list[str] | None = None,
                            adapter_name: str | None = None,
                            **attrs: t.Any) -> t.AsyncGenerator[StreamingResponse, t.Any]:
    if not self.is_ready:
      await self.health()
      if not self.is_ready: raise RuntimeError('Server is not ready. Check server logs for more information.')
    _meta, _config = await self._metadata(), await self._config()
    if llm_config is not None: llm_config = {**_config, **llm_config, **attrs}
    else: llm_config = {**_config, **attrs}
    if _meta['prompt_template'] is not None: prompt = _meta['prompt_template'].format(system_message=_meta['system_message'], instruction=prompt)

    req = Request(prompt=prompt, llm_config=llm_config, stop=stop, adapter_name=adapter_name)
    async with httpx.AsyncClient(base_url=self.address, timeout=self._timeout, **self.client_args) as client:
      async with client.stream('POST', self._build_endpoint('generate_stream'), json=req.model_dump_json(), **self.client_args) as r:
        async for payload in r.aiter_bytes():
          if payload == b'data: [DONE]\n\n': break
          # Skip line
          if payload == b'\n': continue
          if payload.startswith(b'data: '):
            try:
              proc = payload.decode('utf-8').lstrip('data: ').rstrip('\n')
              data = orjson.loads(proc)
            except Exception:
              pass  # FIXME: Handle this
            yield StreamingResponse.from_response_chunk(Response.model_construct(data))
