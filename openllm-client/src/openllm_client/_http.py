from __future__ import annotations

import importlib.metadata, logging, os, typing as t, pydantic

from ._schemas import Helpers, Metadata, Response, StreamingResponse
from ._shim import MAX_RETRIES, AsyncClient, Client

logger = logging.getLogger(__name__)

VERSION = importlib.metadata.version('openllm-client')


def _address_converter(addr: str):
  return addr if '://' in addr else 'http://' + addr


class HTTPClient(Client):
  _helpers: Helpers = pydantic.PrivateAttr()
  _api_version: str = pydantic.PrivateAttr(default='v1')
  _verify: bool = pydantic.PrivateAttr(default=True)
  __metadata: t.Optional[Metadata] = None
  __config: t.Optional[t.Dict[str, t.Any]] = None

  def __repr__(self):
    return f'<HTTPClient address={self.address} timeout={self.timeout} api_version={self._api_version} verify={self._verify}>'

  def __init__(self, address=None, timeout=30, verify=False, max_retries=MAX_RETRIES, api_version='v1'):
    if address is None:
      address = os.getenv('OPENLLM_ENDPOINT')
      if address is None:
        raise ValueError("address must either be provided or through 'OPENLLM_ENDPOINT'")

    super().__init__(_address_converter(address), VERSION, timeout=timeout, max_retries=max_retries)

    self._helpers = Helpers(client=self)
    self._api_version, self._verify = api_version, verify

  @property
  def helpers(self):
    return self._helpers

  def _build_auth_headers(self) -> t.Dict[str, str]:
    env = os.getenv('OPENLLM_AUTH_TOKEN')
    if env is not None:
      return {'Authorization': f'Bearer {env}'}
    return super()._build_auth_headers()

  def __getitem__(self, item):
    if hasattr(self._metadata, item):
      return getattr(self._metadata, item)
    elif item in self._config:
      return self._config[item]
    raise KeyError(f'No attributes: {item}')

  @property
  def _metadata(self):
    if self.__metadata is None:
      path = f'/{self._api_version}/metadata'
      self.__metadata = self._post(path, response_cls=Metadata, json={}, options={'max_retries': self.max_retries})
    return self.__metadata

  @property
  def _config(self) -> dict[str, t.Any]:
    if self.__config is None:
      self.__config = self._metadata.configuration.model_dump()
    return self.__config

  def query(self, prompt, **attrs):
    return self.generate(prompt, **attrs)

  def health(self):
    response = self._get(
      '/readyz', response_cls=None, options={'return_raw_response': True, 'max_retries': self.max_retries}
    )
    return response.status_code == 200

  def generate(
    self, prompt, llm_config=None, stop=None, adapter_name=None, timeout=None, verify=None, **attrs
  ) -> Response:
    if timeout is None:
      timeout = self.timeout
    if verify is None:
      verify = self._verify  # XXX: need to support this again
    if llm_config is not None:
      llm_config = {**self._config, **llm_config, **attrs}
    else:
      llm_config = {**self._config, **attrs}

    if stop is not None:
      if isinstance(stop, str):
        stop = [stop]
      else:
        stop = list(stop)

    return self._post(
      f'/{self._api_version}/generate',
      response_cls=Response,
      json=dict(prompt=prompt, llm_config=llm_config, stop=stop, adapter_name=adapter_name),
      options={'max_retries': self.max_retries},
    )

  def generate_stream(
    self, prompt, llm_config=None, stop=None, adapter_name=None, timeout=None, verify=None, **attrs
  ) -> t.Iterator[StreamingResponse]:
    for response_chunk in self.generate_iterator(prompt, llm_config, stop, adapter_name, timeout, verify, **attrs):
      yield StreamingResponse.from_response_chunk(response_chunk)

  def generate_iterator(
    self, prompt, llm_config=None, stop=None, adapter_name=None, timeout=None, verify=None, **attrs
  ) -> t.Iterator[Response]:
    if timeout is None:
      timeout = self.timeout
    if verify is None:
      verify = self._verify  # XXX: need to support this again
    if llm_config is not None:
      llm_config = {**self._config, **llm_config, **attrs}
    else:
      llm_config = {**self._config, **attrs}

    if stop is not None:
      if isinstance(stop, str):
        stop = [stop]
      else:
        stop = list(stop)

    return self._post(
      f'/{self._api_version}/generate_stream',
      response_cls=Response,
      json=dict(prompt=prompt, llm_config=llm_config, stop=stop, adapter_name=adapter_name),
      options={'max_retries': self.max_retries},
      stream=True,
    )


class AsyncHTTPClient(AsyncClient, pydantic.BaseModel):
  _helpers: Helpers = pydantic.PrivateAttr()
  _api_version: str = pydantic.PrivateAttr(default='v1')
  _verify: bool = pydantic.PrivateAttr(default=True)
  __metadata: t.Optional[Metadata] = None
  __config: t.Optional[t.Dict[str, t.Any]] = None

  def __repr__(self):
    return f'<AsyncHTTPClient address={self.address} timeout={self.timeout} api_version={self._api_version} verify={self._verify}>'

  def __init__(self, address=None, timeout=30, verify=False, max_retries=MAX_RETRIES, api_version='v1'):
    if address is None:
      address = os.getenv('OPENLLM_ENDPOINT')
      if address is None:
        raise ValueError("address must either be provided or through 'OPENLLM_ENDPOINT'")

    super().__init__(_address_converter(address), VERSION, timeout=timeout, max_retries=max_retries)

    # mk messages to be async here
    self._helpers = Helpers.permute(messages=Helpers.async_messages)(async_client=self)
    self._api_version, self._verify = api_version, verify

  @property
  def helpers(self):
    return self._helpers

  def _build_auth_headers(self) -> t.Dict[str, str]:
    env = os.getenv('OPENLLM_AUTH_TOKEN')
    if env is not None:
      return {'Authorization': f'Bearer {env}'}
    return super()._build_auth_headers()

  @property
  async def _metadata(self) -> t.Awaitable[Metadata]:
    if self.__metadata is None:
      self.__metadata = await self._post(
        f'/{self._api_version}/metadata', response_cls=Metadata, json={}, options={'max_retries': self.max_retries}
      )
    return self.__metadata

  @property
  async def _config(self):
    if self.__config is None:
      self.__config = (await self._metadata).configuration.model_dump()
    return self.__config

  async def query(self, prompt, **attrs):
    return await self.generate(prompt, **attrs)

  async def health(self):
    response = await self._get(
      '/readyz', response_cls=None, options={'return_raw_response': True, 'max_retries': self.max_retries}
    )
    return response.status_code == 200

  async def generate(
    self, prompt, llm_config=None, stop=None, adapter_name=None, timeout=None, verify=None, **attrs
  ) -> Response:
    if timeout is None:
      timeout = self.timeout
    if verify is None:
      verify = self._verify  # XXX: need to support this again
    _config = await self._config
    if llm_config is not None:
      llm_config = {**_config, **llm_config, **attrs}
    else:
      llm_config = {**_config, **attrs}

    if stop is not None:
      if isinstance(stop, str):
        stop = [stop]
      else:
        stop = list(stop)

    return await self._post(
      f'/{self._api_version}/generate',
      response_cls=Response,
      json=dict(prompt=prompt, llm_config=llm_config, stop=stop, adapter_name=adapter_name),
      options={'max_retries': self.max_retries},
    )

  async def generate_stream(
    self, prompt, llm_config=None, stop=None, adapter_name=None, timeout=None, verify=None, index=0, **attrs
  ) -> t.AsyncGenerator[StreamingResponse, t.Any]:
    async for response_chunk in self.generate_iterator(
      prompt, llm_config, stop, adapter_name, timeout, verify, **attrs
    ):
      yield StreamingResponse.from_response_chunk(response_chunk, index=index)

  async def generate_iterator(
    self, prompt, llm_config=None, stop=None, adapter_name=None, timeout=None, verify=None, **attrs
  ) -> t.AsyncGenerator[Response, t.Any]:
    if timeout is None:
      timeout = self.timeout
    if verify is None:
      verify = self._verify  # XXX: need to support this again
    _config = await self._config
    if llm_config is not None:
      llm_config = {**_config, **llm_config, **attrs}
    else:
      llm_config = {**_config, **attrs}

    if stop is not None:
      if isinstance(stop, str):
        stop = [stop]
      else:
        stop = list(stop)

    async for response_chunk in await self._post(
      f'/{self._api_version}/generate_stream',
      response_cls=Response,
      json=dict(prompt=prompt, llm_config=llm_config, stop=stop, adapter_name=adapter_name),
      options={'max_retries': self.max_retries},
      stream=True,
    ):
      yield response_chunk
