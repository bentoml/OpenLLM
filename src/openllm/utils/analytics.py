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
"""Telemetry related for OpenLLM tracking.

Users can disable this with OPENLLM_DO_NOT_TRACK envvar.
"""
from __future__ import annotations
import contextlib
import functools
import importlib.metadata
import logging
import os
import re
import typing as t

import attr

import openllm
from bentoml._internal.utils import analytics as _internal_analytics


if t.TYPE_CHECKING:
    from .._types import P
    from .._types import T

logger = logging.getLogger(__name__)


ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}

# This variable is a proxy that will control BENTOML_DO_NOT_TRACK
OPENLLM_DO_NOT_TRACK = "OPENLLM_DO_NOT_TRACK"

DO_NOT_TRACK = os.environ.get(OPENLLM_DO_NOT_TRACK, str(False)).upper()


@functools.lru_cache(maxsize=1)
def do_not_track() -> bool:
    return DO_NOT_TRACK in ENV_VARS_TRUE_VALUES


@functools.lru_cache(maxsize=1)
def _usage_event_debugging() -> bool:
    # For BentoML developers only - debug and print event payload if turned on
    return os.environ.get("__BENTOML_DEBUG_USAGE", str(False)).lower() == "true"


def silent(func: t.Callable[P, T]) -> t.Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
        try:
            return func(*args, **kwargs)
        except Exception as err:
            if _usage_event_debugging():
                if openllm.utils.get_debug_mode():
                    logger.error("Tracking Error: %s", err, stack_info=True, stacklevel=3)
                else:
                    logger.info("Tracking Error: %s", err)
            else:
                logger.debug("Tracking Error: %s", err)

    return wrapper


@silent
def track(event_properties: attr.AttrsInstance) -> None:
    if do_not_track():
        return
    _internal_analytics.track(t.cast("_internal_analytics.schemas.EventMeta", event_properties))


@contextlib.contextmanager
def set_bentoml_tracking() -> t.Generator[None, None, None]:
    original_value = os.environ.pop(_internal_analytics.BENTOML_DO_NOT_TRACK, str(False))
    try:
        os.environ[_internal_analytics.BENTOML_DO_NOT_TRACK] = str(do_not_track())
        yield
    finally:
        os.environ[_internal_analytics.BENTOML_DO_NOT_TRACK] = original_value


class EventMeta:
    @property
    def event_name(self) -> str:
        # camel case to snake case
        event_name = re.sub(r"(?<!^)(?=[A-Z])", "_", self.__class__.__name__).lower()
        # remove "_event" suffix
        suffix_to_remove = "_event"
        if event_name.endswith(suffix_to_remove):
            event_name = event_name[: -len(suffix_to_remove)]
        return event_name


@attr.define
class OpenllmCliEvent(EventMeta):
    cmd_group: str
    cmd_name: str
    openllm_version: str = importlib.metadata.version("openllm")

    # NOTE: reserved for the do_not_track logics
    duration_in_ms: t.Any = attr.field(default=None)
    error_type: str = attr.field(default=None)
    return_code: int = attr.field(default=None)


@attr.define
class StartInitEvent(EventMeta):
    model_name: str
    llm_config: t.Dict[str, t.Any] = attr.field(default=None)

    @staticmethod
    def handler(llm_config: openllm.LLMConfig) -> StartInitEvent:
        return StartInitEvent(model_name=llm_config["model_name"], llm_config=llm_config.model_dump())


def track_start_init(llm_config: openllm.LLMConfig) -> None:
    if do_not_track():
        return
    track(StartInitEvent.handler(llm_config))
