"""The actual client implementation.

Use ``openllm.client`` instead.
This holds the implementation of the client, which is used to communicate with the
OpenLLM server. It is used to send requests to the server, and receive responses.
"""
from __future__ import annotations

from .runtimes.grpc import (
    AsyncGrpcClient as AsyncGrpcClient,
    GrpcClient as GrpcClient,
)
from .runtimes.http import (
    AsyncHTTPClient as AsyncHTTPClient,
    HTTPClient as HTTPClient,
)
