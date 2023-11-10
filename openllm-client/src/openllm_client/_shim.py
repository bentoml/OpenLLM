# This provides a base shim with httpx and acts as base request
from __future__ import annotations
import email.utils
import logging
import platform
import random
import time
import typing as t

import attr
import distro
import httpx

from ._stream import AsyncStream
from ._stream import Response
from ._stream import Stream
from ._typing_compat import Architecture
from ._typing_compat import LiteralString
from ._typing_compat import Platform


logger = logging.getLogger(__name__)

InnerClient = t.TypeVar('InnerClient', bound=t.Union[httpx.Client, httpx.AsyncClient])
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


class JSONData(t.Protocol):
  def model_dump(self) -> t.Dict[str, t.Any]: ...

  def model_dump_json(self) -> str: ...


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
  method: str
  url: str
  params: t.Mapping[str, t.Any]
  json: JSONData
  headers: t.Optional[t.Dict[str, str]]
  max_retries: int = attr.field()

  @max_retries.default
  def _max_retries_default(self) -> int:
    return MAX_RETRIES

  def get_max_retries(self, max_retries: int | None) -> int:
    return max_retries if max_retries is not None else self.max_retries


@attr.define(init=False)
class BaseClient(t.Generic[InnerClient, StreamType]):
  _base_url: httpx.URL = attr.field(converter=_address_converter)
  _version: LiteralVersion
  _timeout: httpx.Timeout
  _max_retries: int
  _inner: InnerClient

  def __init__(
    self,
    *,
    base_url: str | httpx.URL,
    version: str,
    timeout: int | httpx.Timeout = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    client: InnerClient,
    _internal: bool = False,
  ):
    if not _internal:
      raise RuntimeError('Client is reserved to be used internally only.')
    self.__attrs_init__(base_url, version, timeout, max_retries, client)

  def _prepare_url(self, url: str) -> httpx.URL:
    # copied from httpx._merge_url
    merge_url = httpx.URL(url)
    if merge_url.is_relative_url:
      merge_raw = self._base_url.raw_path + merge_url.raw_path.lstrip(b'/')
      return self._base_url.copy_with(path=merge_raw)
    return merge_url

  @property
  def _default_headers(self) -> t.Dict[str, str]:
    return {'Content-Type': 'application/json', 'Accept': 'application/json', **self._platform_headers}

  @property
  def user_agent(self):
    return f'{self.__class__.__name__}/Python {self._version}'

  @property
  def _platform_headers(self):
    return {
      'X-OpenLLM-Client-Package-Version': self._version,
      'User-Agent': self.user_agent,
      'X-OpenLLM-Client-Language': 'Python',
      'X-OpenLLM-Client-Runtime': platform.python_implementation(),
      'X-OpenLLM-Client-Runtime-Version': platform.python_version(),
      'X-OpenLLM-Client-Arch': str(_architecture()),
      'X-OpenLLM-Client-OS': str(_platform()),
    }

  @property
  def base_url(self):
    return self._base_url

  @base_url.setter
  def base_url(self, url):
    self._base_url = url if isinstance(url, httpx.URL) else httpx.URL(url)

  def _remaining_retries(self, remaining_retries: int | None, options: RequestOptions) -> int:
    return remaining_retries if remaining_retries is not None else options.get_max_retries(self._max_retries)

  def _build_headers(self, options: RequestOptions) -> httpx.Headers:
    return httpx.Headers(_merge_mapping(self._default_headers, options.headers or {}))

  def _build_request(self, options: RequestOptions) -> httpx.Request:
    return self._inner.build_request(
      method=options.method, url=self._prepare_url(options.url), json=options.json.model_dump(), params=options.params
    )

  def _calculate_retry_timeout(
    self, remaining_retries: int, options: RequestOptions, headers: t.Optional[httpx.Headers] = None
  ) -> float:
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
    except Exception:
      # omit everything
      retry_after = -1
    if 0 < retry_after <= 60:
      return retry_after  # this is reasonable from users
    initial_delay, max_delay = 0.5, 8.0
    num_retries = max_delay - initial_delay
    sleep = min(initial_delay * pow(2, num_retries), max_delay)  # apply exponential backoff here
    timeout = sleep * (1 - 0.25 * random.random())
    return timeout if timeout >= 0 else 0

  def _should_retry(self, response: httpx.Response) -> bool:
    should_retry_header = response.headers.get('x-should-retry')
    if should_retry_header.lower() == 'true':
      return True
    if should_retry_header.lower() == 'false':
      return False
    if response.status_code == 408:
      return True  # Retry on request timeout
    if response.status_code == 409:
      return True  # Retry on lock timeout
    if response.status_code == 429:
      return True  # Retry on rate limiting
    if response.status_code >= 500:
      return True  # Retry on internal server errors
    return False

  def _process_response_data(self, data: t.Any, response: httpx.Response) -> Response: ...


@attr.define(init=False)
class Client(BaseClient[httpx.Client, Stream[t.Any]]):
  def __init__(
    self,
    base_url: str | httpx.URL,
    version: str,
    timeout: int | httpx.Timeout = DEFAULT_TIMEOUT,
    max_retries: int = MAX_RETRIES,
  ):
    client = httpx.Client(base_url=base_url, timeout=timeout)
    super().__init__(
      base_url=base_url, version=version, timeout=timeout, max_retries=max_retries, client=client, _internal=True
    )

  @property
  def is_closed(self):
    return self._inner.is_closed

  def close(self):
    self._inner.close()

  def __enter__(self):
    return self

  def __exit__(self, *args) -> None:
    self.close()

  def _request(
    self,
    options: RequestOptions,
    remaining_retries: int | None = None,
    *,
    stream: bool = False,
    stream_cls: type[_Stream] | None = None,
  ) -> Response | _Stream:
    retries = self._remaining_retries(remaining_retries, options)
    request = self._build_request(options)
    try:
      response = self._inner.send(request, stream=stream)
      logger.debug(f'HTTP: {request.method} {request.url} {response.status_code}')
      response.raise_for_status()
    except httpx.HTTPStatusError as exc:
      if retries > 0 and self._should_retry(exc.response):
        return self._retry_request(options, retries, exc.response)
      # If the response is streamed then we need to explicitly read the completed response
      exc.response.read()
      raise ValueError(exc.message) from None
    except httpx.TimeoutException:
      if retries > 0:
        return self._retry_request(options, retries, stream=stream, stream_cls=stream_cls)
      raise ValueError(request) from None  # timeout
    except Exception:
      if retries > 0:
        return self._retry_request(options, retries, stream=stream, stream_cls=stream_cls)
      raise ValueError(request) from None  # connection error
    return self._process_response(response, stream=stream, stream_cls=stream_cls)

  def _retry_request(
    self,
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
    return self._request(options, remaining, stream=stream, stream_cls=stream_cls)


@attr.define(init=False)
class AsyncClient(BaseClient[httpx.AsyncClient, AsyncStream[t.Any]]): ...
