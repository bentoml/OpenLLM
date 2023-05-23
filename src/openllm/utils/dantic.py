"""
Code originally derived and adapted from:
https://github.com/samuelcolvin/pydantic/issues/756#issuecomment-798779264.
Credits to Frederik Aalund <https://github.com/frederikaalund> for his valuable suggestions.
"""

from __future__ import annotations

import inspect
import typing as t
from enum import Enum

import click
import orjson
from click import ParamType
from pydantic import BaseModel
from pydantic._internal._utils import lenient_issubclass


def parse_default(default: t.Any, field_type: t.Any) -> t.Any:
    """Converts pydantic defaults into click default types.

    Args:
        default: the current field's default value
        field_type: the type of the current pydantic field

    Returns:
        t.Any: click-compatible default
    """
    # pydantic uses none and ..., click only supports none
    if default in (None, Ellipsis):
        return None
    # for enums we return the name as default
    if lenient_issubclass(field_type, Enum):
        return default.name
    # for modules and such, the name is returned
    if is_typing(field_type):
        module_name = inspect.getmodule(default).__name__
        return f"{module_name}.{default.__name__}"
    # for dictionary types, the default is transformed into string
    if is_mapping(field_type):
        return orjson.dumps(default)
    # for container types, the origin is required
    if is_container(field_type):
        return parse_container_default(default)
    return default


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
    if lenient_issubclass(field_type, t.Mapping):
        return True
    # for everything else or when the typing is more complex, check its origin
    origin = t.get_origin(field_type)
    if origin is None:
        return False
    return lenient_issubclass(origin, t.Mapping)


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
    if lenient_issubclass(field_type, t.Container):
        return True
    origin = t.get_origin(field_type)
    # Early out for non-typing objects
    if origin is None:
        return False
    return lenient_issubclass(origin, t.Container)


def is_typing(field_type: type) -> bool:
    """Checks whether the current type is a module-like type.

    Args:
        field_type (type): pydantic field type

    Returns:
        bool: true if the type is itself a type
    """
    raw = t.get_origin(field_type)
    if raw is None:
        return False
    if raw is type or raw is t.Type:
        return True
    return False


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
    if is_container(arg) or issubclass(arg, BaseModel):
        return JsonType()
    if lenient_issubclass(arg, bytes):
        return BytesType()
    return arg


def parse_container_default(default: t.Any) -> tuple[t.Any, ...] | None:
    """Parses the default type of container types.

    Args:
        default: default type for a container argument.

    Returns:
        tuple[types.Any, ...] | None: JSON version if a pydantic model, else the current default.
    """
    assert issubclass(type(default), t.Sequence)
    return tuple(v.model_validate_json() if isinstance(v, BaseModel) else v for v in default)


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
        if isinstance(value, t.Mapping) or not self.should_load:
            return value
        try:
            return orjson.loads(value)
        except orjson.JSONDecodeError as exc:
            self.fail(f"'{value}' is not a valid JSON string ({str(exc)})", param, ctx)
