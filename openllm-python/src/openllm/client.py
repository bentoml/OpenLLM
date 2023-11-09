"""OpenLLM Python client.

```python
client = openllm.client.HTTPClient("http://localhost:8080")
client.query("What is the difference between gather and scatter?")
```
"""

from __future__ import annotations
import typing as t

import openllm_client


if t.TYPE_CHECKING:
  from openllm_client import AsyncHTTPClient as AsyncHTTPClient
  from openllm_client import HTTPClient as HTTPClient
  # from openllm_client import AsyncGrpcClient as AsyncGrpcClient
  # from openllm_client import GrpcClient as GrpcClient


def __dir__() -> t.Sequence[str]:
  return sorted(dir(openllm_client))


def __getattr__(it: str) -> t.Any:
  return getattr(openllm_client, it)
