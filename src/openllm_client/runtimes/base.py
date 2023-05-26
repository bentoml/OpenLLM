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

import typing as t
from abc import ABC, abstractmethod

import bentoml

import openllm

if t.TYPE_CHECKING:

    class AnnotatedClient(bentoml.client.Client):
        def health(self, *args: t.Any, **kwargs: t.Any) -> t.Any:
            ...

        def generate_v1(self, qa: openllm.GenerationInput) -> dict[str, t.Any]:
            ...

        def metadata_v1(self) -> dict[str, t.Any]:
            ...


class BaseClient(ABC):
    _metadata: dict[str, t.Any]
    _api_version: str
    _config_class: type[bentoml.client.Client]

    _host: str
    _port: str

    __config__: openllm.LLMConfig | None = None
    __client__: AnnotatedClient | None = None

    def __init__(self, address: str, timeout: int = 30):
        self._address = address
        self._timeout = timeout
        assert self._host and self._port, "Make sure to setup _host and _port based on your client implementation."
        self._metadata = self.call("metadata")

    def __init_subclass__(cls, *, client_type: t.Literal["http", "grpc"] = "http", api_version: str = "v1"):
        cls._config_class = bentoml.client.HTTPClient if client_type == "http" else bentoml.client.GrpcClient
        cls._api_version = api_version

    @property
    def model(self) -> str:
        return self._metadata["model_name"]

    @property
    def config(self) -> openllm.LLMConfig:
        if self.__config__ is None:
            self.__config__ = openllm.AutoConfig.for_model(self.model)
        return self.__config__

    def call(self, name: str, *args: t.Any, **kwargs: t.Any) -> t.Any:
        return getattr(self._cached, f"{name}_{self._api_version}")(*args, **kwargs)

    @property
    def _cached(self) -> AnnotatedClient:
        if self.__client__ is None:
            self._config_class.wait_until_server_ready(self._host, int(self._port), timeout=self._timeout)
            self.__client__ = t.cast("AnnotatedClient", self._config_class.from_url(self._address))
        return self.__client__

    def health(self) -> t.Any:
        raise NotImplementedError

    @abstractmethod
    def query(self, prompt: str, **attrs: t.Any) -> dict[str, t.Any] | list[t.Any]:
        raise NotImplementedError

    def chat(self, prompt: str, history: list[str], **attrs: t.Any) -> str:
        raise NotImplementedError
