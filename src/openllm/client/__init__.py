"""The actual client implementation.

Use ``openllm.client`` instead.
This holds the implementation of the client, which is used to communicate with the
OpenLLM server. It is used to send requests to the server, and receive responses.
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
