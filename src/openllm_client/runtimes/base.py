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

import asyncio
import typing as t
from abc import abstractmethod
from urllib.parse import urljoin

import bentoml
import httpx

import openllm

if t.TYPE_CHECKING:

    class AnnotatedClient(bentoml.client.Client):
        def health(self, *args: t.Any, **attrs: t.Any) -> t.Any:
            ...

        async def async_health(self) -> t.Any:
            ...

        def call(self, name: str, inputs: t.Any, **attrs: t.Any) -> t.Any:
            ...

        async def acall(self, name: str, inputs: t.Any, **attrs: t.Any) -> t.Any:
            ...

        def generate_v1(self, qa: openllm.GenerationInput) -> dict[str, t.Any]:
            ...

        def metadata_v1(self) -> dict[str, t.Any]:
            ...


def in_async_context() -> bool:
    try:
        _ = asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


class ClientMixin:
    _api_version: str
    _client_class: type[bentoml.client.Client]

    _host: str
    _port: str

    __client__: AnnotatedClient | None = None
    __llm__: openllm.LLM[t.Any, t.Any] | None = None

    def __init__(self, address: str, timeout: int = 30):
        self._address = address
        self._timeout = timeout
        assert self._host and self._port, "Make sure to setup _host and _port based on your client implementation."

    def __init_subclass__(cls, *, client_type: t.Literal["http", "grpc"] = "http", api_version: str = "v1"):
        cls._client_class = bentoml.client.HTTPClient if client_type == "http" else bentoml.client.GrpcClient
        cls._api_version = api_version

    @property
    def _metadata(self) -> dict[str, t.Any]:
        if in_async_context():
            return httpx.post(urljoin(self._address, f"/{self._api_version}/metadata")).json()
        return self.call("metadata")

    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def framework(self) -> t.Literal["pt", "flax", "tf"]:
        raise NotImplementedError

    @property
    @abstractmethod
    def timeout(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def configuration(self) -> dict[str, t.Any]:
        raise NotImplementedError

    @property
    def llm(self) -> openllm.LLM[t.Any, t.Any]:
        if self.__llm__ is None:
            if self.framework == "flax":
                self.__llm__ = openllm.AutoFlaxLLM.for_model(self.model_name)
            elif self.framework == "tf":
                self.__llm__ = openllm.AutoTFLLM.for_model(self.model_name)
            else:
                self.__llm__ = openllm.AutoLLM.for_model(self.model_name)
        return self.__llm__

    @property
    def config(self) -> openllm.LLMConfig:
        return self.llm.config

    def call(self, name: str, *args: t.Any, **attrs: t.Any) -> t.Any:
        return self._cached.call(f"{name}_{self._api_version}", *args, **attrs)

    async def acall(self, name: str, *args: t.Any, **attrs: t.Any) -> t.Any:
        return await self._cached.async_call(f"{name}_{self._api_version}", *args, **attrs)

    @property
    def _cached(self) -> AnnotatedClient:
        if self.__client__ is None:
            self._client_class.wait_until_server_ready(self._host, int(self._port), timeout=self._timeout)
            self.__client__ = t.cast("AnnotatedClient", self._client_class.from_url(self._address))
        return self.__client__

    def prepare(self, prompt: str, **attrs: t.Any):
        return_raw_response = attrs.pop("return_raw_response", False)
        return return_raw_response, *self.llm.sanitize_parameters(prompt, **attrs)

    @abstractmethod
    def postprocess(self, result: t.Any) -> openllm.GenerationOutput:
        ...


class BaseClient(ClientMixin):
    def health(self) -> t.Any:
        raise NotImplementedError

    @t.overload
    def query(self, prompt: str, *, return_raw_response: t.Literal[False] = ..., **attrs: t.Any) -> str:
        ...

    @t.overload
    def query(self, prompt: str, *, return_raw_response: t.Literal[True] = ..., **attrs: t.Any) -> dict[str, t.Any]:
        ...

    def query(self, prompt: str, **attrs: t.Any) -> dict[str, t.Any] | str:
        return_raw_response, prompt, generate_kwargs, postprocess_kwargs = self.prepare(prompt, **attrs)
        inputs = openllm.GenerationInput(prompt=prompt, llm_config=self.config.model_construct_env(**generate_kwargs))
        if in_async_context():
            result = httpx.post(
                urljoin(self._address, f"/{self._api_version}/generate"),
                json=openllm.utils.bentoml_cattr.unstructure(inputs),
                timeout=self.timeout,
            ).json()
        else:
            result = self.call("generate", inputs)
        r = self.postprocess(result)

        if return_raw_response:
            return openllm.utils.bentoml_cattr.unstructure(r)

        return self.llm.postprocess_generate(prompt, r.responses, **postprocess_kwargs)

    # NOTE: Scikit interface
    def predict(self, prompt: str, **attrs: t.Any) -> t.Any:
        return self.query(prompt, **attrs)

    def chat(self, prompt: str, history: list[str], **attrs: t.Any) -> str:
        raise NotImplementedError


class BaseAsyncClient(ClientMixin):
    async def health(self) -> t.Any:
        raise NotImplementedError

    @t.overload
    async def query(self, prompt: str, *, return_raw_response: t.Literal[False] = ..., **attrs: t.Any) -> str:
        ...

    @t.overload
    async def query(
        self, prompt: str, *, return_raw_response: t.Literal[True] = ..., **attrs: t.Any
    ) -> dict[str, t.Any]:
        ...

    async def query(self, prompt: str, **attrs: t.Any) -> dict[str, t.Any] | str:
        # NOTE: We set use_default_prompt_template to False for now.
        use_default_prompt_template = attrs.pop("use_default_prompt_template", False)
        return_raw_response, prompt, generate_kwargs, postprocess_kwargs = self.prepare(
            prompt, use_default_prompt_template=use_default_prompt_template, **attrs
        )
        inputs = openllm.GenerationInput(prompt=prompt, llm_config=self.config.model_construct_env(**generate_kwargs))
        res = await self.acall("generate", inputs)
        r = self.postprocess(res)

        if return_raw_response:
            return openllm.utils.bentoml_cattr.unstructure(r)

        return self.llm.postprocess_generate(prompt, r.responses, **postprocess_kwargs)

    # NOTE: Scikit interface
    async def predict(self, prompt: str, **attrs: t.Any) -> t.Any:
        return await self.query(prompt, **attrs)

    def chat(self, prompt: str, history: list[str], **attrs: t.Any) -> str:
        raise NotImplementedError
