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

import click
import inflection
import orjson

import bentoml
import openllm

from .. import termui
from .._factory import model_name_argument

if t.TYPE_CHECKING:
    from ..._types import DictStrAny

@click.command("list_models", context_settings=termui.CONTEXT_SETTINGS)
@model_name_argument(required=False)
def cli(model_name: str | None) -> DictStrAny:
    """This is equivalent to openllm models --show-available less the nice table."""
    models = tuple(inflection.dasherize(key) for key in openllm.CONFIG_MAPPING.keys())
    ids_in_local_store = {k: [i for i in bentoml.models.list() if "framework" in i.info.labels and i.info.labels["framework"] == "openllm" and "model_name" in i.info.labels and i.info.labels["model_name"] == k] for k in models}
    if model_name is not None: ids_in_local_store = {k: [i for i in v if "model_name" in i.info.labels and i.info.labels["model_name"] == inflection.dasherize(model_name)] for k,v in ids_in_local_store.items()}
    ids_in_local_store = {k: v for k, v in ids_in_local_store.items() if v}
    local_models = {k: [str(i.tag) for i in val] for k, val in ids_in_local_store.items()}
    termui.echo(orjson.dumps(local_models, option=orjson.OPT_INDENT_2).decode(), fg="white")
    return local_models
