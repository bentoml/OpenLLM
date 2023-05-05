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
Server utilities for OpenLLM. This extends bentoml.Server.

It independently manage processes and threads for runners and servers separately. 
This is an experimental feature and can also be merged to upstream BentoML.
"""
from __future__ import annotations

import functools
import logging
import os
import typing as t

import openllm

logger = logging.getLogger(__name__)


def _start(
    model_name: str,
    framework: t.Literal["flax", "tf", "pt"] | None = None,
    **attrs: t.Any,
):
    _serve_grpc = attrs.pop("_serve_grpc", False)

    if framework is not None:
        os.environ[openllm.utils.FRAMEWORK_ENV_VAR(model_name)] = framework
    openllm.cli.start_model_command(model_name, _serve_grpc=_serve_grpc)(standalone_mode=False, **attrs)


start = functools.partial(_start, _serve_grpc=False)
start_grpc = functools.partial(_start, _serve_grpc=True)

__all__ = ["start", "start_grpc"]
