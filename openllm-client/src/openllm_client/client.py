from __future__ import annotations
import logging

from urllib.parse import urlparse

from ._base import BaseAsyncClient
from ._base import BaseClient

logger = logging.getLogger(__name__)

def process_http_address(self: AsyncHTTPClient | HTTPClient, address: str) -> None:
  address = address if '://' in address else 'http://' + address
  parsed = urlparse(address)
  self._host, *_port = parsed.netloc.split(':')
  if len(_port) == 0: self._port = '80' if parsed.scheme == 'http' else '443'
  else: self._port = next(iter(_port))

class HTTPClient(BaseClient):
  def __init__(self, address: str, timeout: int = 30):
    process_http_address(self, address)
    super().__init__(address, timeout)

class AsyncHTTPClient(BaseAsyncClient):
  def __init__(self, address: str, timeout: int = 30):
    process_http_address(self, address)
    super().__init__(address, timeout)

class GrpcClient(BaseClient):
  def __init__(self, address: str, timeout: int = 30):
    self._host, self._port = address.split(':')
    super().__init__(address, timeout)

class AsyncGrpcClient(BaseAsyncClient):
  def __init__(self, address: str, timeout: int = 30):
    self._host, self._port = address.split(':')
    super().__init__(address, timeout)
