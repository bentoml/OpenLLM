"""OpenLLM Python client.

```python
client = openllm.client.HTTPClient('http://localhost:8080')
client.query('What is the difference between gather and scatter?')
```
"""

from openllm_client import AsyncHTTPClient as AsyncHTTPClient, HTTPClient as HTTPClient
