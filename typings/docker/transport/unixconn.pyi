"""
This type stub file was generated by pyright.
"""

import urllib3
import urllib3.connection
from docker.transport.basehttpadapter import BaseHTTPAdapter

RecentlyUsedContainer = ...

class UnixHTTPConnection(urllib3.connection.HTTPConnection):
    def __init__(self, base_url, unix_socket, timeout=...) -> None: ...
    def connect(self): ...

class UnixHTTPConnectionPool(urllib3.connectionpool.HTTPConnectionPool):
    def __init__(self, base_url, socket_path, timeout=..., maxsize=...) -> None: ...

class UnixHTTPAdapter(BaseHTTPAdapter):
    __attrs__ = ...
    def __init__(self, socket_url, timeout=..., pool_connections=..., max_pool_size=...) -> None: ...
    def get_connection(self, url, proxies=...): ...
    def request_url(self, request, proxies): ...
