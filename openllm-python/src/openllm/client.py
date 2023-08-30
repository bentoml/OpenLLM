'''OpenLLM Python client.

```python
client = openllm.client.HTTPClient("http://localhost:8080")
client.query("What is the difference between gather and scatter?")
```

If the server has embedding supports, use it via `client.embed`:
```python
client.embed("What is the difference between gather and scatter?")
```
'''
from __future__ import annotations
import typing as t

import openllm_client

if t.TYPE_CHECKING:
  from openllm_client import AsyncGrpcClient as AsyncGrpcClient
  from openllm_client import AsyncHTTPClient as AsyncHTTPClient
  from openllm_client import BaseAsyncClient as BaseAsyncClient
  from openllm_client import BaseClient as BaseClient
  from openllm_client import GrpcClient as GrpcClient
  from openllm_client import HTTPClient as HTTPClient

def __dir__() -> t.Sequence[str]:
  return sorted(dir(openllm_client))

def __getattr__(it: str) -> t.Any:
  return getattr(openllm_client, it)
