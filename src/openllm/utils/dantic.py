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
"""A shim provides usable transition from pydantic to attrs."""

from __future__ import annotations

import functools
import os
import typing as t

import attr
import click
import orjson
from click import ParamType

import openllm

if t.TYPE_CHECKING:
    from attr import _ValidatorType  # type: ignore

_T = t.TypeVar("_T")


def _default_converter(value: t.Any, env: str | None) -> t.Any:
    if env is not None:
        value = os.environ.get(env, value)
    if value is not None and isinstance(value, str):
        return eval(value, {"__builtins__": {}}, {})
    return value


def Field(
    default: t.Any = None,
    *,
    ge: int | float | None = None,
    le: int | float | None = None,
    validator: _ValidatorType[_T] | None = None,
    description: str | None = None,
    env: str | None = None,
    **attrs: t.Any,
):
    """A decorator that extends attr.field with additional arguments, which provides the same
    interface as pydantic's Field.

    By default, if both validator and ge are provided, then then ge will be
    piped into first, then all of the other validator will be run afterwards.

    Args:
        ge: Greater than or equal to. Defaults to None.
        docs: the documentation for the field. Defaults to None.
        **kwargs: The rest of the arguments are passed to attr.field
    """
    metadata = attrs.pop("metadata", {})
    if description is None:
        description = "(No description is available)"
    metadata["description"] = description
    if env is not None:
        metadata["env"] = env
    piped: list[_ValidatorType[t.Any]] = []

    converter = attrs.pop("converter", functools.partial(_default_converter, env=env))

    if ge is not None:
        piped.append(attr.validators.ge(ge))
    if le is not None:
        piped.append(attr.validators.le(le))
    if validator is not None:
        piped.append(validator)

    if len(piped) == 0:
        _validator = None
    elif len(piped) == 1:
        _validator = piped[0]
    else:
        _validator = attr.validators.and_(*piped)

    return attr.field(default=default, metadata=metadata, validator=_validator, converter=converter, **attrs)


def allows_multiple(field_type: t.Any) -> bool:
    """Checks whether the current type allows for multiple arguments to be provided as input or not.
    For containers, it exploits click's support for lists and such to use the same option multiple times
    to create a complex object: `python run.py --subsets train --subsets test`
    # becomes `subsets: ["train", "test"]`.
    Args:
        field_type (type): pydantic type

    Returns:
        bool: true if it's a composite field (lists, containers and so on), false otherwise
    """
    # Early out for mappings, since it's better to deal with them using strings.
    if is_mapping(field_type):
        return False
    # Activate multiple option for (simple) container types
    if is_container(field_type):
        args = parse_container_args(field_type)
        # A non-composite type has a single argument, such as 'List[int]'
        # A composite type has a tuple of arguments, like 'Tuple[str, int, int]'.
        # For the moment, only non-composite types are allowed.
        return not isinstance(args, tuple)
    return False


def is_mapping(field_type: type) -> bool:
    """Checks whether this field represents a dictionary or JSON object.

    Args:
        field_type (type): pydantic type

    Returns:
        bool: true when the field is a dict-like object, false otherwise.
    """
    # Early out for standard containers.
    if openllm.utils.lenient_issubclass(field_type, t.Mapping):
        return True
    # for everything else or when the typing is more complex, check its origin
    origin = openllm.utils.get_origin(field_type)
    if origin is None:
        return False
    return openllm.utils.lenient_issubclass(origin, t.Mapping)


def is_container(field_type: type) -> bool:
    """Checks whether the current type is a container type ('contains' other types), like
    lists and tuples.

    Args:
        field_type: pydantic field type

    Returns:
        bool: true if a container, false otherwise
    """
    # do not consider strings or byte arrays as containers
    if field_type in (str, bytes):
        return False
    # Early out for standard containers: list, tuple, range
    if openllm.utils.lenient_issubclass(field_type, t.Container):
        return True
    origin = openllm.utils.get_origin(field_type)
    # Early out for non-typing objects
    if origin is None:
        return False
    return openllm.utils.lenient_issubclass(origin, t.Container)


def parse_container_args(field_type: type) -> ParamType | tuple[ParamType]:
    """Parses the arguments inside a container type (lists, tuples and so on).

    Args:
        field_type: pydantic field type

    Returns:
        ParamType | tuple[ParamType]: single click-compatible type or a tuple
    """
    assert is_container(field_type), "Field type is not a container"
    args = t.get_args(field_type)
    # Early out for untyped containers: standard lists, tuples, List[Any]
    # Use strings when the type is unknown, avoid click's type guessing
    if len(args) == 0:
        return str
    # Early out for homogenous containers: Tuple[int], List[str]
    if len(args) == 1:
        return parse_single_arg(args[0])
    # Early out for homogenous tuples of indefinite length: Tuple[int, ...]
    if len(args) == 2 and args[1] is Ellipsis:
        return parse_single_arg(args[0])
    # Then deal with fixed-length containers: Tuple[str, int, int]
    return tuple(parse_single_arg(arg) for arg in args)


def parse_single_arg(arg: type) -> ParamType:
    """Returns the click-compatible type for container origin types.
    In this case, returns string when it's not inferrable, a JSON for mappings
    and the original type itself in every other case (ints, floats and so on).
    Bytes is a special case, not natively handled by click.

    Args:
        arg (type): single argument

    Returns:
        ParamType: click-compatible type
    """
    # When we don't know the type, we choose 'str'
    if arg is t.Any:
        return str
    # For containers and nested models, we use JSON
    if is_container(arg):
        return JsonType()
    if openllm.utils.lenient_issubclass(arg, bytes):
        return BytesType()
    return arg


class BytesType(ParamType):
    name = "bytes"

    def convert(self, value: t.Any, param: click.Parameter | None, ctx: click.Context | None) -> t.Any:
        if isinstance(value, bytes):
            return value
        try:
            return str.encode(value)
        except Exception as exc:
            self.fail(f"'{value}' is not a valid string ({str(exc)})", param, ctx)


class JsonType(ParamType):
    name = "json"

    def __init__(self, should_load: bool = True) -> None:
        super().__init__()
        self.should_load = should_load

    def convert(self, value: t.Any, param: click.Parameter | None, ctx: click.Context | None) -> t.Any:
        if openllm.utils.LazyType[t.Mapping[str, str]](t.Mapping).isinstance(value) or not self.should_load:
            return value
        try:
            return orjson.loads(value)
        except orjson.JSONDecodeError as exc:
            self.fail(f"'{value}' is not a valid JSON string ({str(exc)})", param, ctx)
