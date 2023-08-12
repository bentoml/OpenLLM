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

from .runtimes import (
    AsyncGrpcClient as AsyncGrpcClient,
    AsyncHTTPClient as AsyncHTTPClient,
    BaseAsyncClient as BaseAsyncClient,
    BaseClient as BaseClient,
    GrpcClient as GrpcClient,
    HTTPClient as HTTPClient,
)
