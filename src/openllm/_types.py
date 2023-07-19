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
"""Types definition for OpenLLM.

Note that this module SHOULD NOT BE IMPORTED DURING RUNTIME, as this serve only for typing purposes.
It will raises a RuntimeError if this is imported eagerly.
"""
from __future__ import annotations
import typing as t


if not t.TYPE_CHECKING:
    raise RuntimeError(f"{__name__} should not be imported during runtime")


import attr

import bentoml
from bentoml._internal.types import ModelSignatureDict as ModelSignatureDict

from ._configuration import AdapterType
from ._configuration import LiteralRuntime as LiteralRuntime


if t.TYPE_CHECKING:
    import click
    import peft
    import torch

    import openllm
    from bentoml._internal.runner.runnable import RunnableMethod
    from bentoml._internal.runner.runner import RunnerMethod
    from bentoml._internal.runner.strategy import Strategy


AnyCallable = t.Callable[..., t.Any]
DictStrAny = dict[str, t.Any]
ListAny = list[t.Any]
ListStr = list[str]
TupleAny = tuple[t.Any, ...]
P = t.ParamSpec("P")
O_co = t.TypeVar("O_co", covariant=True)
T = t.TypeVar("T")
Ts = t.TypeVarTuple("Ts")
At = t.TypeVar("At", bound=attr.AttrsInstance)


class ClickFunctionWrapper(t.Protocol[P, O_co]):
    __name__: str
    __click_params__: list[click.Option]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> O_co:
        ...


# F is a t.Callable[P, O_co] with compatible to ClickFunctionWrapper
class F(t.Generic[P, O_co]):
    __name__: str
    __click_params__: list[click.Option]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> O_co:
        ...


_MT = t.TypeVar("_MT", covariant=True)


class _StubsMixin(t.Generic[_MT], t.Protocol):
    def save_pretrained(self, save_directory: str, **kwargs: t.Any) -> t.Any:
        ...

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: t.Any, **kwargs: t.Any) -> _MT:
        ...


class ModelProtocol(_StubsMixin[_MT], t.Protocol):
    @property
    def framework(self) -> str:
        ...


class TokenizerProtocol(_StubsMixin[_MT], t.Protocol):
    @t.override
    def save_pretrained(self, save_directory: str, **kwargs: t.Any) -> tuple[str]:
        ...


class PeftAdapterOutput(t.TypedDict):
    success: bool
    result: dict[str, peft.PeftConfig]
    error_msg: str


class LLMEmbeddings(t.TypedDict):
    embeddings: torch.Tensor
    num_tokens: int


class AdaptersTuple(TupleAny):
    adapter_id: str
    name: str | None
    config: DictStrAny


AdaptersMapping = dict[AdapterType, tuple[AdaptersTuple, ...]]


class LLMRunnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("amd.com/gpu", "nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True

    model: ModelProtocol[t.Any]

    set_adapter: RunnableMethod[LLMRunnable, [str], dict[t.Literal["success", "error_msg"], bool | str]]
    __call__: RunnableMethod[LLMRunnable, [str], list[t.Any]]
    embeddings: RunnableMethod[LLMRunnable, [list[str]], LLMEmbeddings]
    generate: RunnableMethod[LLMRunnable, [str], list[t.Any]]
    generate_one: RunnableMethod[LLMRunnable, [str, list[str]], t.Sequence[dict[t.Literal["generated_text"], str]]]
    generate_iterator: RunnableMethod[LLMRunnable, [str], t.Generator[t.Any, None, None]]


class LLMRunner(bentoml.Runner):
    __doc__: str
    __module__: str
    llm_type: str
    identifying_params: dict[str, t.Any]
    llm: openllm.LLM[t.Any, t.Any]
    model: ModelProtocol[t.Any]
    config: openllm.LLMConfig

    embeddings: RunnerMethod[LLMRunnable, [list[str]], LLMEmbeddings]
    generate: RunnerMethod[LLMRunnable, [str], list[t.Any]]
    generate_one: RunnerMethod[LLMRunnable, [str, list[str]], t.Sequence[dict[t.Literal["generated_text"], str]]]
    generate_iterator: RunnerMethod[LLMRunnable, [str], t.Generator[t.Any, None, None]]

    def __init__(
        self,
        runnable_class: type[LLMRunnable],
        *,
        runnable_init_params: dict[str, t.Any] | None = ...,
        name: str | None = ...,
        scheduling_strategy: type[Strategy] = ...,
        models: list[bentoml.Model] | None = ...,
        max_batch_size: int | None = ...,
        max_latency_ms: int | None = ...,
        method_configs: dict[str, dict[str, int]] | None = ...,
        embedded: bool = False,
    ) -> None:
        ...

    def __call__(self, prompt: str, **attrs: t.Any) -> t.Any:
        ...

    def embed(self, prompt: str | list[str]) -> LLMEmbeddings:
        ...

    def run(self, prompt: str, **attrs: t.Any) -> t.Any:
        ...

    async def async_run(self, prompt: str, **attrs: t.Any) -> t.Any:
        ...

    def download_model(self) -> bentoml.Model:
        ...

    @property
    def peft_adapters(self) -> PeftAdapterOutput:
        ...

    @property
    def __repr_keys__(self) -> set[str]:
        ...
