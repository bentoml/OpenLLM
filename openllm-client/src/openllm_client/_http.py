from __future__ import annotations
import typing as t

from urllib.parse import urlparse

import attr
import httpx

if t.TYPE_CHECKING:
  import aiohttp

def _address_validator(_: t.Any, attr: attr.Attribute[t.Any], value: str) -> None:
  if not isinstance(value, str): raise TypeError(f'{attr.name} must be a string')
  if not urlparse(value).netloc: raise ValueError(f'{attr.name} must be a valid URL')

@attr.define
class HTTPClient:
  address: str = attr.field(validator=_address_validator)
  timeout: int = 30
  api_version: str = 'v1'
  client_args: t.Dict[str, t.Any] = attr.field(factory=dict)
  _inner: httpx.Client = attr.field(init=False)

  def __attrs_post_init__(self) -> None:
    self._inner = httpx.Client(base_url=self.address, timeout=self.timeout, **self.client_args)

  def health(self):
    return self._inner.get('/readyz')

  def query(self, prompt: str, **attrs: t.Any):
    prompt, generate_kwargs, postprocess_kwargs = self.sanitize_parameters(prompt, **attrs)

  def __del__(self) -> None:
    self._inner.close()

@attr.define
class AsyncHTTPClient:
  address: str
  timeout: int = 30
  api_version: str = 'v1'
  _inner: aiohttp.ClientSession = attr.field(init=False)
