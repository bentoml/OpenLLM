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
"""
Types definition for OpenLLM.

Note that this module SHOULD NOT BE IMPORTED DURING RUNTIME, as this serve only for typing purposes.
It will raises a RuntimeError if this is imported eagerly.
"""
from __future__ import annotations

import typing as t


if not t.TYPE_CHECKING:
    raise RuntimeError(f"{__name__} should not be imported during runtime")

import click
import bentoml
import openllm


P = t.ParamSpec("P")
O_co = t.TypeVar("O_co", covariant=True)


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


PeftAdapterOutput = dict[t.Literal["success", "result", "error_msg"], bool | str | dict[t.Any, t.Any]]


class LLMRunner(bentoml.Runner):
    __doc__: str
    __module__: str
    model: ModelProtocol[t.Any]
    llm: openllm.LLM[t.Any, t.Any]
    config: openllm.LLMConfig
    llm_type: str
    identifying_params: dict[str, t.Any]

    def __call__(self, *args: t.Any, **attrs: t.Any) -> t.Any:
        ...

    def download_model(self, quiet: bool = ...) -> None:
        ...

    @property
    def peft_adapters(self) -> PeftAdapterOutput:
        ...
