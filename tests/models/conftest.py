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
import contextlib
import functools
import logging
import sys
import time
import typing as t
from abc import ABC
from abc import abstractmethod

import attr
import docker
import docker.errors
import docker.types
import orjson
import pytest
from syrupy.extensions.json import JSONSnapshotExtension

import openllm
from openllm._llm import normalise_model_name


logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    import subprocess

    from openllm_client.runtimes.base import BaseAsyncClient
    from syrupy.assertion import SnapshotAssertion
    from syrupy.types import PropertyFilter
    from syrupy.types import PropertyMatcher
    from syrupy.types import SerializableData
    from syrupy.types import SerializedData

    from openllm._configuration import GenerationConfig
    from openllm._types import DictStrAny
    from openllm._types import ListAny

else:
    DictStrAny = dict
    ListAny = list


class ResponseComparator(JSONSnapshotExtension):
    def serialize(
        self,
        data: SerializableData,
        *,
        exclude: PropertyFilter | None = None,
        matcher: PropertyMatcher | None = None,
    ) -> SerializedData:
        if openllm.utils.LazyType(ListAny).isinstance(data):
            data = [d.unmarshaled for d in data]
        else:
            data = data.unmarshaled
        data = self._filter(data=data, depth=0, path=(), exclude=exclude, matcher=matcher)
        return orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode()

    def matches(self, *, serialized_data: SerializableData, snapshot_data: SerializableData) -> bool:
        def convert_data(data: SerializableData) -> openllm.GenerationOutput | t.Sequence[openllm.GenerationOutput]:
            try:
                data = orjson.loads(data)
            except orjson.JSONDecodeError as err:
                raise ValueError(f"Failed to decode JSON data: {data}") from err
            if openllm.utils.LazyType(DictStrAny).isinstance(data):
                return openllm.GenerationOutput(**data)
            elif openllm.utils.LazyType(ListAny).isinstance(data):
                return [openllm.GenerationOutput(**d) for d in data]
            else:
                raise NotImplementedError(f"Data {data} has unsupported type.")

        serialized_data = convert_data(serialized_data)
        snapshot_data = convert_data(snapshot_data)

        if openllm.utils.LazyType(ListAny).isinstance(serialized_data):
            serialized_data = [serialized_data]
        if openllm.utils.LazyType(ListAny).isinstance(snapshot_data):
            snapshot_data = [snapshot_data]

        def eq_config(s: GenerationConfig, t: GenerationConfig) -> bool:
            return s == t

        def eq_output(s: openllm.GenerationOutput, t: openllm.GenerationOutput) -> bool:
            return (
                len(s.responses) == len(t.responses)
                and all([_s == _t for _s, _t in zip(s.responses, t.responses)])
                and eq_config(s.marshaled_config, t.marshaled_config)
            )

        return len(serialized_data) == len(snapshot_data) and all(
            [eq_output(s, t) for s, t in zip(serialized_data, snapshot_data)]
        )


@pytest.fixture()
def response_snapshot(snapshot: SnapshotAssertion):
    return snapshot.use_extension(ResponseComparator)


@attr.define(init=False)
class _Handle(ABC):
    port: int
    deployment_mode: t.Literal["container", "local"]

    client: BaseAsyncClient[t.Any] = attr.field(init=False)

    if t.TYPE_CHECKING:

        def __attrs_init__(self, *args: t.Any, **attrs: t.Any):
            ...

    def __attrs_post_init__(self):
        self.client = openllm.client.AsyncHTTPClient(f"http://localhost:{self.port}")

    @abstractmethod
    def status(self) -> bool:
        raise NotImplementedError

    async def health(self, timeout: int = 240):
        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.status():
                raise RuntimeError(f"Failed to initialise {self.__class__.__name__}")
            await self.client.health()
            try:
                await self.client.query("sanity")
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"Handle failed to initialise within {timeout} seconds.")


@attr.define(init=False)
class LocalHandle(_Handle):
    process: subprocess.Popen[bytes]

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        port: int,
        deployment_mode: t.Literal["container", "local"],
    ):
        self.__attrs_init__(port, deployment_mode, process)

    def status(self) -> bool:
        return self.process.poll() is None


class HandleProtocol(t.Protocol):
    @contextlib.contextmanager
    def __call__(
        *,
        model: str,
        model_id: str,
        image_tag: str,
        quantize: t.AnyStr | None = None,
    ) -> t.Generator[_Handle, None, None]:
        ...


@attr.define(init=False)
class DockerHandle(_Handle):
    container_name: str
    docker_client: docker.DockerClient

    def __init__(
        self,
        docker_client: docker.DockerClient,
        container_name: str,
        port: int,
        deployment_mode: t.Literal["container", "local"],
    ):
        self.__attrs_init__(port, deployment_mode, container_name, docker_client)

    def status(self) -> bool:
        container = self.docker_client.containers.get(self.container_name)
        return container.status in ["running", "created"]


@contextlib.contextmanager
def _local_handle(
    model: str,
    model_id: str,
    image_tag: str,
    deployment_mode: t.Literal["container", "local"],
    quantize: t.Literal["int8", "int4", "gptq"] | None = None,
    *,
    _serve_grpc: bool = False,
):
    with openllm.utils.reserve_free_port() as port:
        pass

    if not _serve_grpc:
        proc = openllm.start(
            model, model_id=model_id, quantize=quantize, additional_args=["--port", str(port)], __test__=True
        )
    else:
        proc = openllm.start_grpc(
            model, model_id=model_id, quantize=quantize, additional_args=["--port", str(port)], __test__=True
        )

    yield LocalHandle(proc, port, deployment_mode)
    proc.terminate()
    proc.wait(60)

    process_output = proc.stdout.read()
    print(process_output, file=sys.stderr)

    proc.stdout.close()
    if proc.stderr:
        proc.stderr.close()


@contextlib.contextmanager
def _container_handle(
    model: str,
    model_id: str,
    image_tag: str,
    deployment_mode: t.Literal["container", "local"],
    quantize: t.Literal["int8", "int4", "gptq"] | None = None,
    *,
    _serve_grpc: bool = False,
):
    envvar = openllm.utils.EnvVarMixin(model)

    with openllm.utils.reserve_free_port() as port, openllm.utils.reserve_free_port() as prom_port:
        pass
    container_name = f"openllm-{model}-{normalise_model_name(model_id)}".replace("-", "_")
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        container.stop()
        container.wait()
        container.remove()
    except docker.errors.NotFound:
        pass

    args = ["serve" if not _serve_grpc else "serve-grpc"]

    env: DictStrAny = {}

    if quantize is not None:
        env[envvar.quantize] = quantize

    gpus = openllm.utils.device_count() or -1
    devs = [docker.types.DeviceRequest(count=gpus, capabilities=[["gpu"]])] if gpus > 0 else None

    container = client.containers.run(
        image_tag,
        command=args,
        name=container_name,
        environment=env,
        auto_remove=False,
        detach=True,
        device_requests=devs,
        ports={"3000/tcp": port, "3001/tcp": prom_port},
    )

    yield DockerHandle(client, container.name, port, deployment_mode)

    try:
        container.stop()
        container.wait()
    except docker.errors.NotFound:
        pass

    container_output = container.logs().decode("utf-8")
    print(container_output, file=sys.stderr)

    container.remove()


@pytest.fixture(scope="session", autouse=True)
def clean_context() -> t.Generator[contextlib.ExitStack, None, None]:
    stack = contextlib.ExitStack()
    yield stack
    stack.close()


@pytest.fixture(scope="module")
def el() -> t.Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(params=["container", "local"], scope="session")
def deployment_mode(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def handler(el: asyncio.AbstractEventLoop, deployment_mode: t.Literal["container", "local"]):
    if deployment_mode == "container":
        return functools.partial(_container_handle, deployment_mode=deployment_mode)
    elif deployment_mode == "local":
        return functools.partial(_local_handle, deployment_mode=deployment_mode)
    else:
        raise ValueError(f"Unknown deployment mode: {deployment_mode}")
