"""OpenLLM Python client.

```python
client = openllm.client.HTTPClient("http://localhost:8080")
client.query("What is the difference between gather and scatter?")
```

If the server has embedding supports, use it via `client.embed`:
```python
client.embed("What is the difference between gather and scatter?")
```
"""
from __future__ import annotations
import openllm_client, typing as t
if t.TYPE_CHECKING: from openllm_client import AsyncHTTPClient as AsyncHTTPClient, BaseAsyncClient as BaseAsyncClient, BaseClient as BaseClient, HTTPClient as HTTPClient, GrpcClient as GrpcClient, AsyncGrpcClient as AsyncGrpcClient
def __dir__() -> t.Sequence[str]:
  return sorted(dir(openllm_client))
def __getattr__(it: str) -> t.Any:
  return getattr(openllm_client, it)
