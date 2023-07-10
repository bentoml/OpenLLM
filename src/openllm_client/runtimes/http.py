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

from __future__ import annotations
import logging
import typing as t
from urllib.parse import urlparse

import orjson

import openllm

from .base import BaseAsyncClient
from .base import BaseClient


if t.TYPE_CHECKING:
    from openllm._types import DictStrAny
    from openllm._types import LiteralRuntime
else:
    DictStrAny = dict


logger = logging.getLogger(__name__)


class HTTPClientMixin:
    if t.TYPE_CHECKING:

        @property
        def _metadata(self) -> DictStrAny:
            ...

    @property
    def model_name(self) -> str:
        try:
            return self._metadata["model_name"]
        except KeyError:
            raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None

    @property
    def model_id(self) -> str:
        try:
            return self._metadata["model_name"]
        except KeyError:
            raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None

    @property
    def framework(self) -> LiteralRuntime:
        try:
            return self._metadata["framework"]
        except KeyError:
            raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None

    @property
    def timeout(self) -> int:
        try:
            return self._metadata["timeout"]
        except KeyError:
            raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None

    @property
    def configuration(self) -> dict[str, t.Any]:
        try:
            return orjson.loads(self._metadata["configuration"])
        except KeyError:
            raise RuntimeError("Malformed service endpoint. (Possible malicious)") from None

    def postprocess(self, result: dict[str, t.Any]) -> openllm.GenerationOutput:
        return openllm.GenerationOutput(**result)


class HTTPClient(HTTPClientMixin, BaseClient[DictStrAny]):
    def __init__(self, address: str, timeout: int = 30):
        address = address if "://" in address else "http://" + address
        self._host, self._port = urlparse(address).netloc.split(":")
        super().__init__(address, timeout)

    def health(self) -> t.Any:
        return self._cached.health()


class AsyncHTTPClient(HTTPClientMixin, BaseAsyncClient[DictStrAny]):
    def __init__(self, address: str, timeout: int = 30):
        address = address if "://" in address else "http://" + address
        self._host, self._port = urlparse(address).netloc.split(":")
        super().__init__(address, timeout)

    async def health(self) -> t.Any:
        return await self._cached.async_health()
