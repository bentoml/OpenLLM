# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenLLM client.

To start interact with the server, you can do the following:

>>> import openllm
>>> client = openllm.client.HTTPClient("http://localhost:3000")
>>> client.query("What is the meaning of life?")
"""
from __future__ import annotations
import importlib
import itertools
import typing as t


_import_structure: dict[str, list[str]] = {
    "runtimes.grpc": ["AsyncGrpcClient", "GrpcClient"],
    "runtimes.http": ["AsyncHTTPClient", "HTTPClient"],
}

if t.TYPE_CHECKING:
    from openllm_client import AsyncGrpcClient as AsyncGrpcClient
    from openllm_client import AsyncHTTPClient as AsyncHTTPClient
    from openllm_client import GrpcClient as GrpcClient
    from openllm_client import HTTPClient as HTTPClient

__all__ = list(itertools.chain.from_iterable(_import_structure.values()))

_module = "openllm_client"


def __dir__() -> list[str]:
    return sorted(__all__)


def __getattr__(name: str) -> t.Any:
    if name in _import_structure:
        return importlib.import_module(f".{name}", _module)
    try:
        module = next(module for module, attrs in _import_structure.items() if name in attrs)
    except StopIteration:
        raise AttributeError(f"module {_module} has no attribute {name}") from None
    return getattr(importlib.import_module(f".{module}", _module), name)
