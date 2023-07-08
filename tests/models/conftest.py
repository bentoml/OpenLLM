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

import sys
import types
import attr
import functools
import time
import logging
import asyncio
import subprocess
import docker
import docker.errors
import docker.types
import contextlib
from abc import ABC
from abc import abstractmethod
import typing as t

import pytest
import orjson

import openllm
from openllm._llm import normalise_model_name
import itertools
from syrupy.extensions.json import JSONSnapshotExtension

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from openllm_client.runtimes.base import BaseAsyncClient
    from openllm._types import DictStrAny
    from openllm._types import ListAny
    from openllm._configuration import GenerationConfig
    from syrupy.assertion import SnapshotAssertion

    from syrupy.types import (
        PropertyFilter,
        PropertyMatcher,
        SerializableData,
        SerializedData,
    )
    from openllm._types import LiteralRuntime

else:
    DictStrAny = dict
    ListAny = list


_FRAMEWORK_MAPPING = {"flan_t5": "google/flan-t5-small", "opt": "facebook/opt-125m"}
_PROMPT_MAPPING = {
    "qa": "Answer the following yes/no question by reasoning step-by-step. Can you write a whole Haiku in a single tweet?",
    "default": "What is the weather in SF?",
}


def parametrise_local_llm(
    model: str,
) -> t.Generator[tuple[openllm.LLMRunner | openllm.LLM[t.Any, t.Any], str], None, None]:
    if model not in _FRAMEWORK_MAPPING:
        pytest.skip(f"'{model}' is not yet supported in framework testing.")

    runtime_impl: tuple[LiteralRuntime, ...] = tuple()
    if model in openllm.MODEL_MAPPING_NAMES:
        runtime_impl += ("pt",)
    if model in openllm.MODEL_FLAX_MAPPING_NAMES:
        runtime_impl += ("flax",)
    if model in openllm.MODEL_TF_MAPPING_NAMES:
        runtime_impl += ("tf",)

    for framework, prompt in itertools.product(runtime_impl, _PROMPT_MAPPING.keys()):
        llm, runner_kwargs = openllm.infer_auto_class(framework).for_model(
            model, model_id=_FRAMEWORK_MAPPING[model], ensure_available=True, return_runner_kwargs=True
        )
        yield llm, prompt
        runner = llm.to_runner(**runner_kwargs)
        runner.init_local(quiet=True)
        yield runner, prompt


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    model = t.cast(types.ModuleType, metafunc.module).__name__.split(".")[-1].strip("test_")
    if "prompt" in metafunc.fixturenames and "llm" in metafunc.fixturenames:
        metafunc.parametrize("prompt,llm", [(p, llm) for p, llm in parametrise_local_llm(model)])


def convert_data(data: SerializableData) -> openllm.GenerationOutput | t.Sequence[openllm.GenerationOutput]:
    try:
        data = orjson.loads(data)
    except orjson.JSONDecodeError:
        raise ValueError(f"Failed to decode JSON data: {data}")
    if openllm.utils.LazyType(DictStrAny).isinstance(data):
        return openllm.GenerationOutput(**data)
    elif openllm.utils.LazyType(ListAny).isinstance(data):
        return [openllm.GenerationOutput(**d) for d in data]
    else:
        raise NotImplementedError(f"Data {data} has unsupported type.")


class ResponseComparator(JSONSnapshotExtension):
    def serialize(
        self,
        data: SerializableData,
        *,
        exclude: PropertyFilter | None = None,
        matcher: PropertyMatcher | None = None,
    ) -> SerializedData:
        data = self._filter(data=data, depth=0, path=(), exclude=exclude, matcher=matcher)
        return orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS)

    def matches(self, *, serialized_data: SerializableData, snapshot_data: SerializableData) -> bool:
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


@pytest.fixture(scope="module", name="response_snapshot")
def fixture_response_snapshot(snapshot: SnapshotAssertion):
    snapshot.use_extension(ResponseComparator)


@attr.define
class _Handle(ABC):
    port: int
    timeout: int = attr.field(default=60)

    client: BaseAsyncClient = attr.field(init=False)

    def __attrs_post_init__(self):
        self.client = openllm.client.AsyncHTTPClient(f"http://localhost:{self.port}")

    @abstractmethod
    def status(self) -> bool:
        raise NotImplementedError

    async def health(self, timeout: int | None = None):
        if timeout is None:
            timeout = self.timeout

        start_time = time.time()
        while time.time() - start_time < timeout:
            if not self.status():
                raise RuntimeError(f"Failed to initialise {self.__class__.__name__}")
            try:
                await self.client.query("sanity")
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError(f"Handle failed to initialise within {self.timeout} seconds.")


@attr.define(init=False)
class LocalHandle(_Handle):
    process: subprocess.Popen[bytes]

    def __init__(self, process: subprocess.Popen[bytes], port: int, timeout: int = 60):
        super().__init__(port=port, timeout=timeout)
        self.process = process

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

    def __init__(self, docker_client: docker.DockerClient, container_name: str, port: int, timeout: int = 60):
        super().__init__(port=port, timeout=timeout)
        self.docker_client = docker_client
        self.container_name = container_name

    def status(self) -> bool:
        container = self.docker_client.containers.get(self.container_name)
        return container.status in ["running", "created"]


@contextlib.contextmanager
def _local_handle(
    model: str,
    model_id: str,
    image_tag: str,
    quantize: t.Literal["int8", "int4", "gptq"] | None = None,
    *,
    clean_context: contextlib.ExitStack,
    _serve_grpc: bool = False,
):
    port = clean_context.enter_context(openllm.utils.reserve_free_port())
    if not _serve_grpc:
        proc = openllm.start(
            model, model_id=model_id, quantize=quantize, additional_args=["--port", str(port)], __test__=True
        )
    else:
        proc = openllm.start_grpc(
            model, model_id=model_id, quantize=quantize, additional_args=["--port", str(port)], __test__=True
        )

    yield LocalHandle(proc, port)
    proc.terminate()
    proc.wait(60)

    process_output = proc.stdout.read().decode("utf-8")
    print(process_output, file=sys.stderr)

    proc.stdout.close()
    proc.stderr.close()
    clean_context.close()


@contextlib.contextmanager
def _container_handle(
    model: str,
    model_id: str,
    image_tag: str,
    quantize: t.Literal["int8", "int4", "gptq"] | None = None,
    *,
    clean_context: contextlib.ExitStack,
    _serve_grpc: bool = False,
):
    envvar = openllm.utils.EnvVarMixin(model)

    port = clean_context.enter_context(openllm.utils.reserve_free_port())
    container_name = f"openllm-{model}-{normalise_model_name(model_id)}"
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        container.stop()
        container.wait()
    except docker.errors.NotFound:
        pass

    args = ["serve" if not _serve_grpc else "serve-grpc"]

    env: DictStrAny = {}

    if quantize is not None:
        env[envvar.quantize] = quantize

    available = openllm.utils.gpu_count()
    gpus = len(available) if len(available) > 0 else 1

    container = client.containers.run(
        image_tag,
        command=args,
        name=container_name,
        environment=env,
        auto_remove=False,
        detach=True,
        device_requests=[docker.types.DeviceRequest(count=gpus, capabilities=[["gpu"]])],
        ports={"80/tcp": port},
    )

    yield DockerHandle(client, container_name, port)

    try:
        container.stop()
        container.wait()
    except docker.errors.NotFound:
        pass

    container_output = container.logs().decode("utf-8")
    print(container_output, file=sys.stderr)

    container.remove()


@pytest.fixture(scope="session", autouse=True, name="clean_context")
def fixture_clean_context() -> t.Generator[contextlib.ExitStack, None, None]:
    stack = contextlib.ExitStack()
    yield stack
    stack.close()


@pytest.fixture(scope="module", name="el")
def fixture_el() -> t.Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest.fixture(
    name="deployment_mode",
    params=["container", "local"],
    scope="session",
)
def fixture_deployment_mode(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module", name="handler")
def fixture_handler(
    el: asyncio.AbstractEventLoop,
    clean_context: contextlib.ExitStack,
    deployment_mode: t.Literal["container", "local"],
):
    if deployment_mode == "container":
        return functools.partial(_container_handle, clean_context=clean_context)
    elif deployment_mode == "local":
        return functools.partial(_local_handle, clean_context=clean_context)
    else:
        raise ValueError(f"Unknown deployment mode: {deployment_mode}")
