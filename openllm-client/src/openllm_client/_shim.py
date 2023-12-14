# This provides a base shim with httpx and acts as base request
from __future__ import annotations
import asyncio
import email.utils
import logging
import platform
import random
import time
import typing as t

import anyio
import attr
import distro
import httpx

from ._stream import AsyncStream, Response, Stream
from ._typing_compat import Architecture, LiteralString, Platform
from ._utils import converter

logger = logging.getLogger(__name__)

InnerClient = t.TypeVar('InnerClient', bound=t.Union[httpx.Client, httpx.AsyncClient])
StreamType = t.TypeVar('StreamType', bound=t.Union[Stream[t.Any], AsyncStream[t.Any]])
_Stream = t.TypeVar('_Stream', bound=Stream[t.Any])
_AsyncStream = t.TypeVar('_AsyncStream', bound=AsyncStream[t.Any])
LiteralVersion = t.Annotated[LiteralString, t.Literal['v1'], str]


def _address_converter(addr: str | httpx.URL) -> httpx.URL:
  if isinstance(addr, httpx.URL):
    url = addr
  else:
    url = httpx.URL(addr if '://' in addr else f'http://{addr}')
  if not url.raw_path.endswith(b'/'):
    url = url.copy_with(path=url.raw_path + b'/')
  return url


MAX_RETRIES = 2
DEFAULT_TIMEOUT = httpx.Timeout(5.0)  # Similar to httpx


_T_co = t.TypeVar('_T_co', covariant=True)
_T = t.TypeVar('_T')


def _merge_mapping(a: t.Mapping[_T_co, _T], b: t.Mapping[_T_co, _T]) -> t.Dict[_T_co, _T]:
  # does the merging and filter out None
  return {k: v for k, v in {**a, **b}.items() if v is not None}


def _platform() -> Platform:
  system = platform.system().lower()
  platform_name = platform.platform().lower()
  if system == 'darwin':
    return 'MacOS'
  elif system == 'windows':
    return 'Windows'
  elif system == 'linux':
    distro_id = distro.id()
    if distro_id == 'freebsd':
      return 'FreeBSD'
    elif distro_id == 'openbsd':
      return 'OpenBSD'
    else:
      return 'Linux'
  elif 'android' in platform_name:
    return 'Android'
  elif 'iphone' in platform_name:
    return 'iOS'
  elif 'ipad' in platform_name:
    return 'iPadOS'
  if platform_name:
    return f'Other:{platform_name}'
  return 'Unknown'


def _architecture() -> Architecture:
  machine = platform.machine().lower()
  if machine in {'arm64', 'aarch64'}:
    return 'arm64'
  elif machine in {'arm', 'aarch32'}:
    return 'arm'
  elif machine in {'x86_64', 'amd64'}:
    return 'x86_64'
  elif machine in {'x86', 'i386', 'i686'}:
    return 'x86'
  elif machine:
    return f'Other:{machine}'
  return 'Unknown'


@t.final
@attr.frozen(auto_attribs=True)
class RequestOptions:
  method: str = attr.field(converter=str.lower)
  url: str
  json: t.Optional[t.Dict[str, t.Any]] = attr.field(default=None)
  params: t.Optional[t.Mapping[str, t.Any]] = attr.field(default=None)
  headers: t.Optional[t.Dict[str, str]] = attr.field(default=None)
  max_retries: int = attr.field(default=MAX_RETRIES)
  return_raw_response: bool = attr.field(default=False)

  def get_max_retries(self, max_retries: int | None) -> int:
    return max_retries if max_retries is not None else self.max_retries

  @classmethod
  def model_construct(cls, **options: t.Any) -> RequestOptions:
    return cls(**options)


@t.final
@attr.frozen(auto_attribs=True)
class APIResponse(t.Generic[Response]):
  _raw_response: httpx.Response
  _client: BaseClient[t.Any, t.Any]
  _response_cls: type[Response]
  _stream: bool
  _stream_cls: t.Optional[t.Union[t.Type[Stream[t.Any]], t.Type[AsyncStream[t.Any]]]]
  _options: RequestOptions

  _parsed: t.Optional[Response] = attr.field(default=None, repr=False)

  def parse(self):
    if self._options.return_raw_response:
      return self._raw_response
    if self._parsed is not None:
      return self._parsed
    if self._stream:
      stream_cls = self._stream_cls or self._client._default_stream_cls
      return stream_cls(response_cls=self._response_cls, response=self._raw_response, client=self._client)

    if self._response_cls is str:
      return self._raw_response.text

    content_type, *_ = self._raw_response.headers.get('content-type', '').split(';')
    if content_type != 'application/json':
      # Since users specific different content_type, then we return the raw binary text without and deserialisation
      return self._raw_response.text

    data = self._raw_response.json()
    try:
      return self._client._process_response_data(data=data, response_cls=self._response_cls, raw_response=self._raw_response)
    except Exception as exc:
      raise ValueError(exc) from None  # validation error here

  @property
  def headers(self):
    return self._raw_response.headers

  @property
  def status_code(self):
    return self._raw_response.status_code

  @property
  def request(self):
    return self._raw_response.request

  @property
  def url(self):
    return self._raw_response.url

  @property
  def content(self):
    return self._raw_response.content

  @property
  def text(self):
    return self._raw_response.text

  @property
  def http_version(self):
    return self._raw_response.http_version


@attr.define(init=False)
class BaseClient(t.Generic[InnerClient, StreamType]):
  _base_url: httpx.URL = attr.field(converter=_address_converter)
  _version: LiteralVersion
  _timeout: httpx.Timeout = attr.field(converter=httpx.Timeout)
  _max_retries: int
  _inner: InnerClient
  _default_stream_cls: t.Type[StreamType]
  _auth_headers: t.Dict[str, str] = attr.field(init=False)

  def __init__(
    self,
    *,
    base_url: str | httpx.URL,
    version: str,
    timeout: int | httpx.Timeout = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    client: InnerClient,
    _default_stream_cls: t.Type[StreamType],
    _internal: bool = False,
  ):
    if not _internal:
      raise RuntimeError('Client is reserved to be used internally only.')
    self.__attrs_init__(base_url, version, timeout, max_retries, client, _default_stream_cls)
    self._auth_headers = self._build_auth_headers()

  def _prepare_url(self, url: str) -> httpx.URL:
    # copied from httpx._merge_url
    merge_url = httpx.URL(url)
    if merge_url.is_relative_url:
      merge_raw = self._base_url.raw_path + merge_url.raw_path.lstrip(b'/')
      return self._base_url.copy_with(raw_path=merge_raw)
    return merge_url

  @property
  def is_closed(self):
    return self._inner.is_closed

  @property
  def is_ready(self):
    return not self.is_closed  # backward compat

  @property
  def base_url(self):
    return self._base_url

  @property
  def address(self):
    return str(self.base_url)

  @base_url.setter
  def base_url(self, url):
    self._base_url = url if isinstance(url, httpx.URL) else httpx.URL(url)

  def _build_auth_headers(self) -> t.Dict[str, str]:
    return {}  # can be overridden for subclasses for auth support

  @property
  def auth(self) -> httpx.Auth | None:
    return None

  @property
  def user_agent(self):
    return f'{self.__class__.__name__}/Python {self._version}'

  @property
  def auth_headers(self):
    return self._auth_headers

  @property
  def _default_headers(self) -> t.Dict[str, str]:
    return {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'User-Agent': self.user_agent,
      **self.platform_headers,
      **self.auth_headers,
    }

  @property
  def platform_headers(self):
    return {
      'X-OpenLLM-Client-Package-Version': self._version,
      'X-OpenLLM-Client-Language': 'Python',
      'X-OpenLLM-Client-Runtime': platform.python_implementation(),
      'X-OpenLLM-Client-Runtime-Version': platform.python_version(),
      'X-OpenLLM-Client-Arch': str(_architecture()),
      'X-OpenLLM-Client-OS': str(_platform()),
    }

  def _remaining_retries(self, remaining_retries: int | None, options: RequestOptions) -> int:
    return remaining_retries if remaining_retries is not None else options.get_max_retries(self._max_retries)

  def _build_headers(self, options: RequestOptions) -> httpx.Headers:
    return httpx.Headers(_merge_mapping(self._default_headers, options.headers or {}))

  def _build_request(self, options: RequestOptions) -> httpx.Request:
    return self._inner.build_request(
      method=options.method, headers=self._build_headers(options), url=self._prepare_url(options.url), json=options.json, params=options.params
    )

  def _calculate_retry_timeout(self, remaining_retries: int, options: RequestOptions, headers: t.Optional[httpx.Headers] = None) -> float:
    max_retries = options.get_max_retries(self._max_retries)
    # https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Retry-After
    try:
      if headers is not None:
        retry_header = headers.get('retry-after')
        try:
          retry_after = int(retry_header)
        except ValueError:
          tup = email.utils.parsedate_tz(retry_header)
          if tup is None:
            retry_after = -1
          else:
            retry_after = int(email.utils.mktime_tz(tup) - time.time())
      else:
        retry_after = -1
    except Exception:
      # omit everything
      retry_after = -1
    if 0 < retry_after <= 60:
      return retry_after  # this is reasonable from users
    initial_delay, max_delay = 0.5, 8.0
    num_retries = max_retries - remaining_retries

    sleep = min(initial_delay * pow(2.0, num_retries), max_delay)  # apply exponential backoff here
    timeout = sleep * (1 - 0.25 * random.random())
    return timeout if timeout >= 0 else 0

  def _should_retry(self, response: httpx.Response) -> bool:
    should_retry_header = response.headers.get('x-should-retry')
    if should_retry_header:
      if should_retry_header.lower() == 'true':
        return True
      if should_retry_header.lower() == 'false':
        return False
    if response.status_code in {408, 409, 429}:
      return True
    if response.status_code >= 500:
      return True
    return False

  def _process_response_data(self, *, response_cls: type[Response], data: t.Dict[str, t.Any], raw_response: httpx.Response) -> Response:
    return converter.structure(data, response_cls)

  def _process_response(
    self,
    *,
    response_cls: type[Response],
    options: RequestOptions,
    raw_response: httpx.Response,
    stream: bool,
    stream_cls: type[_Stream] | type[_AsyncStream] | None,
  ) -> Response:
    return APIResponse(
      raw_response=raw_response, client=self, response_cls=response_cls, stream=stream, stream_cls=stream_cls, options=options
    ).parse()


@attr.define(init=False)
class Client(BaseClient[httpx.Client, Stream[t.Any]]):
  def __init__(self, base_url: str | httpx.URL, version: str, timeout: int | httpx.Timeout = DEFAULT_TIMEOUT, max_retries: int = MAX_RETRIES):
    super().__init__(
      base_url=base_url,
      version=version,
      timeout=timeout,
      max_retries=max_retries,
      client=httpx.Client(base_url=base_url, timeout=timeout),
      _default_stream_cls=Stream,
      _internal=True,
    )

  def close(self):
    self._inner.close()

  def __enter__(self):
    return self

  def __exit__(self, *args) -> None:
    self.close()

  def __del__(self):
    self.close()

  def request(
    self,
    response_cls: type[Response],
    options: RequestOptions,
    remaining_retries: t.Optional[int] = None,
    *,
    stream: bool = False,
    stream_cls: type[_Stream] | None = None,
  ) -> Response | _Stream:
    return self._request(response_cls=response_cls, options=options, remaining_retries=remaining_retries, stream=stream, stream_cls=stream_cls)

  def _request(
    self,
    response_cls: type[Response],
    options: RequestOptions,
    remaining_retries: int | None = None,
    *,
    stream: bool = False,
    stream_cls: type[_Stream] | None = None,
  ) -> Response | _Stream:
    retries = self._remaining_retries(remaining_retries, options)
    request = self._build_request(options)
    try:
      response = self._inner.send(request, auth=self.auth, stream=stream)
      logger.debug('HTTP [%s, %s]: %s [%i]', request.method, request.url, response.status_code, response.reason_phrase)
      response.raise_for_status()
    except httpx.HTTPStatusError as exc:
      if retries > 0 and self._should_retry(exc.response):
        return self._retry_request(response_cls, options, retries, exc.response.headers, stream=stream, stream_cls=stream_cls)
      # If the response is streamed then we need to explicitly read the completed response
      exc.response.read()
      raise ValueError(exc.message) from None
    except httpx.TimeoutException:
      if retries > 0:
        return self._retry_request(response_cls, options, retries, stream=stream, stream_cls=stream_cls)
      raise ValueError(request) from None  # timeout
    except Exception:
      if retries > 0:
        return self._retry_request(response_cls, options, retries, stream=stream, stream_cls=stream_cls)
      raise ValueError(request) from None  # connection error

    return self._process_response(response_cls=response_cls, options=options, raw_response=response, stream=stream, stream_cls=stream_cls)

  def _retry_request(
    self,
    response_cls: type[Response],
    options: RequestOptions,
    remaining_retries: int,
    response_headers: httpx.Headers | None = None,
    *,
    stream: bool = False,
    stream_cls: type[_Stream] | None,
  ) -> Response | _Stream:
    remaining = remaining_retries - 1
    timeout = self._calculate_retry_timeout(remaining_retries, options, response_headers)
    logger.info('Retrying request to %s in %f seconds', options.url, timeout)
    # In synchronous thread we are blocking the thread. Depends on how users want to do this downstream.
    time.sleep(timeout)
    return self._request(response_cls, options, remaining, stream=stream, stream_cls=stream_cls)

  def _get(
    self,
    path: str,
    *,
    response_cls: type[Response],
    options: dict[str, t.Any] | None = None,
    stream: bool = False,
    stream_cls: type[_Stream] | None = None,
  ) -> Response | _Stream:
    if options is None:
      options = {}
    return self.request(response_cls, RequestOptions(method='GET', url=path, **options), stream=stream, stream_cls=stream_cls)

  def _post(
    self,
    path: str,
    *,
    response_cls: type[Response],
    json: dict[str, t.Any],
    options: dict[str, t.Any] | None = None,
    stream: bool = False,
    stream_cls: type[_Stream] | None = None,
  ) -> Response | _Stream:
    if options is None:
      options = {}
    return self.request(response_cls, RequestOptions(method='POST', url=path, json=json, **options), stream=stream, stream_cls=stream_cls)


@attr.define(init=False)
class AsyncClient(BaseClient[httpx.AsyncClient, AsyncStream[t.Any]]):
  def __init__(self, base_url: str | httpx.URL, version: str, timeout: int | httpx.Timeout = DEFAULT_TIMEOUT, max_retries: int = MAX_RETRIES):
    super().__init__(
      base_url=base_url,
      version=version,
      timeout=timeout,
      max_retries=max_retries,
      client=httpx.AsyncClient(base_url=base_url, timeout=timeout),
      _default_stream_cls=AsyncStream,
      _internal=True,
    )

  async def close(self):
    await self._inner.aclose()

  async def __aenter__(self):
    return self

  async def __aexit__(self, *args) -> None:
    await self.close()

  def __del__(self):
    try:
      loop = asyncio.get_event_loop()
      if loop.is_running():
        loop.create_task(self.close())
      else:
        loop.run_until_complete(self.close())
    except Exception:
      pass

  async def request(
    self,
    response_cls: type[Response],
    options: RequestOptions,
    remaining_retries: t.Optional[int] = None,
    *,
    stream: bool = False,
    stream_cls: type[_AsyncStream] | None = None,
  ) -> Response | _AsyncStream:
    return await self._request(response_cls, options, remaining_retries=remaining_retries, stream=stream, stream_cls=stream_cls)

  async def _request(
    self,
    response_cls: type[Response],
    options: RequestOptions,
    remaining_retries: int | None = None,
    *,
    stream: bool = False,
    stream_cls: type[_AsyncStream] | None = None,
  ) -> Response | _AsyncStream:
    retries = self._remaining_retries(remaining_retries, options)
    request = self._build_request(options)

    try:
      response = await self._inner.send(request, auth=self.auth, stream=stream)
      logger.debug('HTTP [%s, %s]: %s [%i]', request.method, request.url, response.status_code, response.reason_phrase)
      response.raise_for_status()
    except httpx.HTTPStatusError as exc:
      if retries > 0 and self._should_retry(exc.response):
        return self._retry_request(response_cls, options, retries, exc.response.headers, stream=stream, stream_cls=stream_cls)
      # If the response is streamed then we need to explicitly read the completed response
      await exc.response.aread()
      raise ValueError(exc.message) from None
    except httpx.ConnectTimeout as err:
      if retries > 0:
        return await self._retry_request(response_cls, options, retries, stream=stream, stream_cls=stream_cls)
      raise ValueError(request) from err  # timeout
    except httpx.ReadTimeout:
      # We don't retry on ReadTimeout error, so something might happen on server-side
      raise
    except httpx.TimeoutException as err:
      if retries > 0:
        return await self._retry_request(response_cls, options, retries, stream=stream, stream_cls=stream_cls)
      raise ValueError(request) from err  # timeout
    except Exception as err:
      if retries > 0:
        return await self._retry_request(response_cls, options, retries, stream=stream, stream_cls=stream_cls)
      raise ValueError(request) from err  # connection error

    return self._process_response(response_cls=response_cls, options=options, raw_response=response, stream=stream, stream_cls=stream_cls)

  async def _retry_request(
    self,
    response_cls: type[Response],
    options: RequestOptions,
    remaining_retries: int,
    response_headers: httpx.Headers | None = None,
    *,
    stream: bool,
    stream_cls: type[_AsyncStream] | None,
  ):
    remaining = remaining_retries - 1
    timeout = self._calculate_retry_timeout(remaining_retries, options, response_headers)
    logger.info('Retrying request to %s in %f seconds', options.url, timeout)
    await anyio.sleep(timeout)
    return await self._request(response_cls, options, remaining, stream=stream, stream_cls=stream_cls)

  async def _get(
    self,
    path: str,
    *,
    response_cls: type[Response],
    options: dict[str, t.Any] | None = None,
    stream: bool = False,
    stream_cls: type[_AsyncStream] | None = None,
  ) -> Response | _AsyncStream:
    if options is None:
      options = {}
    return await self.request(response_cls, RequestOptions(method='GET', url=path, **options), stream=stream, stream_cls=stream_cls)

  async def _post(
    self,
    path: str,
    *,
    response_cls: type[Response],
    json: dict[str, t.Any],
    options: dict[str, t.Any] | None = None,
    stream: bool = False,
    stream_cls: type[_AsyncStream] | None = None,
  ) -> Response | _AsyncStream:
    if options is None:
      options = {}
    return await self.request(response_cls, RequestOptions(method='POST', url=path, json=json, **options), stream=stream, stream_cls=stream_cls)
