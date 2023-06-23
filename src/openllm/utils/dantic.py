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
import importlib
import os
import typing as t
from enum import Enum

import attr
import click
import click_option_group as cog
import inflection
import orjson
from click import ParamType

import openllm


# NOTE: We need to do this so that overload can register
# correct overloads to typing registry
if hasattr(t, "get_overloads"):
    from typing import overload
else:
    from typing_extensions import overload

if t.TYPE_CHECKING:
    from attr import _ValidatorType

    from .._types import ClickFunctionWrapper
    from .._types import F
    from .._types import O_co
    from .._types import P

_T = t.TypeVar("_T")


@overload
def attrs_to_options(
    name: str,
    field: attr.Attribute[t.Any],
    model_name: str,
    typ: type[t.Any] | None = None,
    suffix_generation: bool = False,
) -> F[..., F[..., openllm.LLMConfig]]:
    ...


@overload
def attrs_to_options(  # type: ignore (overlapping overload)
    name: str,
    field: attr.Attribute[O_co],
    model_name: str,
    typ: type[t.Any] | None = None,
    suffix_generation: bool = False,
) -> F[..., F[P, O_co]]:
    ...


def attrs_to_options(
    name: str,
    field: attr.Attribute[t.Any],
    model_name: str,
    typ: type[t.Any] | None = None,
    suffix_generation: bool = False,
) -> t.Callable[..., ClickFunctionWrapper[..., t.Any]]:
    # TODO: support parsing nested attrs class and Union
    envvar = field.metadata["env"]
    dasherized = inflection.dasherize(name)
    underscored = inflection.underscore(name)

    if typ in (None, attr.NOTHING):
        typ = field.type

    full_option_name = f"--{dasherized}"
    if field.type is bool:
        full_option_name += f"/--no-{dasherized}"
    if suffix_generation:
        identifier = f"{model_name}_generation_{underscored}"
    else:
        identifier = f"{model_name}_{underscored}"

    return cog.optgroup.option(
        identifier,
        full_option_name,
        type=parse_type(typ),
        required=field.default is attr.NOTHING,
        default=field.default if field.default not in (attr.NOTHING, None) else None,
        show_default=True,
        multiple=allows_multiple(typ),
        help=field.metadata.get("description", "(No description provided)"),
        show_envvar=True,
        envvar=envvar,
    )


def env_converter(value: t.Any, env: str | None = None) -> t.Any:
    if env is not None:
        value = os.environ.get(env, value)
    if value is not None and isinstance(value, str):
        try:
            return orjson.loads(value.lower())
        except orjson.JSONDecodeError as err:
            raise RuntimeError(f"Failed to parse '{value}' from '{env}': {err}")
    return value


def Field(
    default: t.Any = None,
    *,
    ge: int | float | None = None,
    le: int | float | None = None,
    validator: _ValidatorType[_T] | None = None,
    description: str | None = None,
    env: str | None = None,
    auto_default: bool = False,
    use_default_converter: bool = True,
    **attrs: t.Any,
):
    """A decorator that extends attr.field with additional arguments, which provides the same
    interface as pydantic's Field.

    By default, if both validator and ge are provided, then then ge will be
    piped into first, then all of the other validator will be run afterwards.

    Args:
        ge: Greater than or equal to. Defaults to None.
        description: the documentation for the field. Defaults to None.
        env: the environment variable to read from. Defaults to None.
        auto_default: a bool indicating whether to use the default value as the environment.
            Defaults to False. If set to True, the behaviour of this Field will also depends
            on kw_only. If kw_only=True, the this field will become 'Required' and the default
            value is omitted. If kw_only=False, then the default value will be used as before.
        use_default_converter: a bool indicating whether to use the default converter. Defaults
            to True. If set to False, then the default converter will not be used. The default
            converter converts a given value from the environment variable for this given Field.
        **kwargs: The rest of the arguments are passed to attr.field
    """
    metadata = attrs.pop("metadata", {})
    if description is None:
        description = "(No description provided)"
    metadata["description"] = description
    if env is not None:
        metadata["env"] = env
    piped: list[_ValidatorType[t.Any]] = []

    converter = attrs.pop("converter", None)
    if use_default_converter:
        converter = functools.partial(env_converter, env=env)

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

    factory = attrs.pop("factory", None)
    if factory is not None and default is not None:
        raise RuntimeError("'factory' and 'default' are mutually exclusive.")
    # NOTE: the behaviour of this is we will respect factory over the default
    if factory is not None:
        attrs["factory"] = factory
    else:
        attrs["default"] = default

    kw_only = attrs.pop("kw_only", False)
    if auto_default and kw_only:
        attrs.pop("default")

    return attr.field(metadata=metadata, validator=_validator, converter=converter, **attrs)


def parse_type(field_type: t.Any) -> ParamType | tuple[ParamType]:
    """Transforms the pydantic field's type into a click-compatible type.

    Args:
        field_type: pydantic field type

    Returns:
        ParamType: click type equivalent
    """
    from . import lenient_issubclass

    assert t.get_origin(field_type) is not t.Union, "Unions are not supported"
    # enumeration strings or other Enum derivatives
    if lenient_issubclass(field_type, Enum):
        return EnumChoice(enum=field_type, case_sensitive=True)
    # literals are enum-like with way less functionality
    if is_literal(field_type):
        return LiteralChoice(enum=field_type, case_sensitive=True)
    # modules, classes, functions
    if is_typing(field_type):
        return ModuleType()
    # entire dictionaries:
    # using a Dict, convert in advance
    if is_mapping(field_type):
        return JsonType()
    # list, List[p], Tuple[p], Set[p] and so on
    if is_container(field_type):
        return parse_container_args(field_type)
    # bytes are not natively supported by click
    if lenient_issubclass(field_type, bytes):
        return BytesType()
    # return the current type: it should be a primitive
    return field_type


def is_typing(field_type: type) -> bool:
    """Checks whether the current type is a module-like type.

    Args:
        field_type: pydantic field type

    Returns:
        bool: true if the type is itself a type
    """
    raw = t.get_origin(field_type)
    if raw is None:
        return False
    if raw is type or raw is t.Type:
        return True
    return False


def is_literal(field_type: type) -> bool:
    """Checks whether the given field type is a Literal type or not.
    Literals are weird: isinstance and subclass do not work, so you compare
    the origin with the Literal declaration itself.

    Args:
        field_type: current pydantic type

    Returns:
        bool: true if Literal type, false otherwise
    """
    origin = t.get_origin(field_type)
    return origin is not None and origin is t.Literal


class ModuleType(ParamType):
    name = "module"

    def _import_object(self, value: str) -> t.Any:
        module_name, class_name = value.rsplit(".", maxsplit=1)
        assert all(s.isidentifier() for s in module_name.split(".")), f"'{value}' is not a valid module name"
        assert class_name.isidentifier(), f"Variable '{class_name}' is not a valid identifier"

        module = importlib.import_module(module_name)
        if class_name:
            try:
                return getattr(module, class_name)
            except AttributeError:
                raise ImportError(f"Module '{module_name}' does not define a '{class_name}' variable.")
        return None

    def convert(self, value: str, param: click.Parameter | None, ctx: click.Context | None) -> t.Any:
        try:
            if isinstance(value, str):
                return self._import_object(value)
            return value
        except Exception as exc:
            self.fail(f"'{value}' is not a valid object ({type(exc)}: {str(exc)})", param, ctx)


class EnumChoice(click.Choice):
    name = "enum"

    def __init__(self, enum: Enum, case_sensitive: bool = False):
        self.mapping = enum
        self.internal_type = enum
        super().__init__([e.name for e in self.mapping], case_sensitive)

    def convert(self, value: t.Any, param: click.Parameter | None, ctx: click.Context | None) -> Enum:
        if isinstance(value, self.internal_type):
            return value
        result = super().convert(value, param, ctx)
        if isinstance(result, str):
            result = self.mapping[result]
        return result


class LiteralChoice(EnumChoice):
    name = "literal"

    def __init__(self, enum: t.LiteralString, case_sensitive: bool = False):
        # expect every literal value to belong to the same primitive type
        values = list(enum.__args__)
        item_type = type(values[0])
        assert all(isinstance(v, item_type) for v in values), f"Field {enum} contains items of different types"
        self.internal_type = item_type
        self.mapping = {str(v): v for v in values}
        super(EnumChoice, self).__init__(list(self.mapping.keys()), case_sensitive)


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
    origin = t.get_origin(field_type)
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
    origin = t.get_origin(field_type)
    # Early out for non-typing objects
    if origin is None:
        return False
    return openllm.utils.lenient_issubclass(origin, t.Container)


def parse_container_args(field_type: type[t.Any]) -> ParamType | tuple[ParamType]:
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
