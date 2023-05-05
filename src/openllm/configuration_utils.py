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
Configuration utilities for OpenLLM. All model configuration will inherit from openllm.configuration_utils.LLMConfig.

Note that ``openllm.configuration_utils.LLMConfig`` is a subclass of ``pydantic.BaseModel``. It also has a ``to_cli``
that returns a list of Click-compatible options for the model. such options will then be parsed to ``openllm.__main__.cli``.
"""
from __future__ import annotations

import typing as t
from abc import ABC

import click
import pydantic
import yaml
from bentoml._internal.models.model import ModelSignature
from click_option_group import optgroup
from clidantic.click import (allows_multiple, parse_default, parse_type,
                             should_show_default)
from clidantic.convert import param_from_field
from pydantic.utils import lenient_issubclass

import openllm

if t.TYPE_CHECKING:
    from openllm.types import F, P


__all__ = ["LLMConfig", "ModelSignature"]

delimiter = "."
internal_delimiter = "__"


def models_to_options(
    model: type[LLMConfig], parent_path: tuple[str, ...] = tuple()
) -> t.Generator[t.Callable[[F[P]], F[P]], None, None]:
    # The following logics are inspired from clidantic.convert
    for field in model.__fields__.values():
        kebab_name = field.name.replace("_", "-")
        assert internal_delimiter not in kebab_name, f"Field name {kebab_name} contains internal delimiter"
        if lenient_issubclass(field.outer_type_, pydantic.BaseModel):
            yield from models_to_options(field.outer_type_, parent_path=parent_path + (kebab_name,))
            continue

        params = param_from_field(field, kebab_name, delimiter, internal_delimiter, parent_path)
        yield optgroup.option(
            *params,
            type=parse_type(field.outer_type_),
            required=field.required,
            default=parse_default(field.default, field.outer_type_),
            show_default=should_show_default(field.default, field.outer_type_),
            multiple=allows_multiple(field.outer_type_),
            help=field.field_info.description,
            show_envvar=True,
            envvar=f"OPENLLM_{model.__module__.rsplit('.', maxsplit=2)[-2].upper()}_{field.name.upper()}",
        )


class BaseConfig(pydantic.BaseModel, ABC):
    class Config:
        extra = "forbid"
        underscore_attrs_are_private = True

    def with_options(self, **kwargs: t.Any) -> BaseConfig:
        return self.copy(update=kwargs)

    @classmethod
    def from_yaml(cls, path: str, ctx: str | None = None) -> BaseConfig:
        with open(openllm.utils.resolve_user_filepath(path, ctx=ctx), "rb") as f:
            content = yaml.safe_load(f)
        return cls(**content)

    def to_yaml(self):
        return yaml.safe_dump(self.dict(), sort_keys=False)


class LLMConfig(BaseConfig):
    @staticmethod
    def generate_click_options(config: LLMConfig) -> t.Callable[[t.Callable[..., t.Any]], click.Command]:
        klass = config.__class__
        group = optgroup.group(f"{klass.__name__} options", help=f"[Auto-generated from '{klass.__qualname__}']")

        def wrapper(f: t.Callable[..., t.Any]) -> click.Command:
            for option in reversed(list(models_to_options(klass))):
                f = option(f)
            return group(f)

        return wrapper
