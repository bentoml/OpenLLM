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
Telemetry related for OpenLLM tracking.

Users can disable this with OPENLLM_DO_NOT_TRACK envvar.
"""
from __future__ import annotations

import contextlib
import functools
import os
import typing as t
from datetime import datetime

import attr
import bentoml
from bentoml._internal.utils import analytics as _internal_analytics
from bentoml._internal.utils.analytics import usage_stats as _internal_usage

if t.TYPE_CHECKING:
    import openllm
    import click

from ..__about__ import __version__

ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

# This variable is a proxy that will control BENTOML_DO_NOT_TRACK
OPENLLM_DO_NOT_TRACK = "OPENLLM_DO_NOT_TRACK"

DO_NOT_TRACK = os.environ.get(OPENLLM_DO_NOT_TRACK, str(False)).upper()


@functools.lru_cache(maxsize=1)
def do_not_track() -> bool:
    return DO_NOT_TRACK in ENV_VARS_TRUE_VALUES


@_internal_usage.silent
def track(event_properties: _internal_analytics.schemas.EventMeta):
    if do_not_track():
        return
    _internal_analytics.track(event_properties)


@contextlib.contextmanager
def set_bentoml_tracking():
    original_value = os.environ.pop(_internal_analytics.BENTOML_DO_NOT_TRACK, str(False))
    try:
        os.environ[_internal_analytics.BENTOML_DO_NOT_TRACK] = str(do_not_track())
        yield
    finally:
        os.environ[_internal_analytics.BENTOML_DO_NOT_TRACK] = original_value


@attr.define
class OpenllmCliEvent(_internal_analytics.schemas.EventMeta):
    cmd_group: str
    cmd_name: str
    openllm_version: str = __version__

    # NOTE: reserved for the do_not_track logics
    duration_in_ms: t.Any = attr.field(default=None)
    error_type: str = attr.field(default=None)
    return_code: int = attr.field(default=None)


if t.TYPE_CHECKING:
    T_con = t.TypeVar("T_con", contravariant=True)

    class HandlerProtocol(t.Protocol[T_con]):
        @staticmethod
        def __call__(group: click.Group, cmd_name: str, return_value: T_con | None = None) -> OpenllmCliEvent:
            ...


@attr.define
class StartInitEvent(_internal_analytics.schemas.EventMeta):
    model_name: str
    supported_gpu: bool = attr.field(default=False)
    llm_config: t.Dict[str, t.Any] = attr.field(default=None)

    @staticmethod
    def handler(
        llm_config: openllm.LLMConfig,
        supported_gpu: bool = False,
    ) -> StartInitEvent:
        return StartInitEvent(
            model_name=llm_config.__openllm_model_name__,
            llm_config=llm_config.model_dump(),
            supported_gpu=supported_gpu,
        )


def track_start_init(
    llm_config: openllm.LLMConfig,
    supported_gpu: bool = False,
):
    if do_not_track():
        return
    track(StartInitEvent.handler(llm_config, supported_gpu))


@attr.define
class BuildEvent(OpenllmCliEvent):
    bento_creation_timestamp: datetime = attr.field(default=None)
    bento_size_in_gb: float = attr.field(default=0)
    model_size_in_gb: float = attr.field(default=0)
    model_type: str = attr.field(default=None)
    model_framework: str = attr.field(default=None)

    @staticmethod
    def handler(group: click.Group, cmd_name: str, return_value: bentoml.Bento | None = None) -> BuildEvent:
        from bentoml._internal.utils import calc_dir_size

        assert group.name is not None, "group name should not be None"
        if return_value is not None:
            bento = return_value
            return BuildEvent(
                group.name,
                cmd_name,
                bento_creation_timestamp=bento.info.creation_time,
                bento_size_in_gb=calc_dir_size(bento.path) / 1024**3,
                model_size_in_gb=calc_dir_size(bento.path_of("/models")) / 1024**3,
                model_type=bento.info.labels["_type"],
                model_framework=bento.info.labels["_framework"],
            )
        return BuildEvent(group.name, cmd_name)


cli_events_map: dict[str, dict[str, HandlerProtocol[t.Any]]] = {
    "openllm": {"build": BuildEvent.handler, "bundle": BuildEvent.handler}
}
