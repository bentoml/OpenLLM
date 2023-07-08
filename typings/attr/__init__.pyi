from __future__ import annotations

import enum
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import ParamSpec
from typing import Protocol
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TypeGuard
from typing import TypeVar
from typing import Union
from typing import overload

from . import converters as converters
from . import exceptions as exceptions
from . import filters as filters
from . import setters as setters
from . import validators as validators
from ._cmp import cmp_using as cmp_using
from ._typing_compat import AttrsInstance_
from ._version_info import VersionInfo

__version__: str
__version_info__: VersionInfo
__title__: str
__description__: str
__url__: str
__uri__: str
__author__: str
__email__: str
__license__: str
__copyright__: str
_T = TypeVar("_T")
_C = TypeVar("_C", bound=type)
_P = ParamSpec("_P")
_EqOrderType = Union[bool, Callable[[Any], Any]]
_ValidatorType = Callable[[Any, "Attribute[_T]", _T], Any]
_ConverterType = Callable[[Any], Any]
_FilterType = Callable[["Attribute[_T]", _T], bool]
_ReprType = Callable[[Any], str]
_ReprArgType = Union[bool, _ReprType]
_OnSetAttrType = Callable[[Any, "Attribute[Any]", Any], Any]
_OnSetAttrArgType = Union[_OnSetAttrType, List[_OnSetAttrType], setters._NoOpType]
_FieldTransformer = Callable[[type, List["Attribute[Any]"]], List["Attribute[Any]"]]
_ValidatorArgType = Union[_ValidatorType[_T], Sequence[_ValidatorType[_T]]]

class ReprProtocol(Protocol):
    def __call__(__self, self: Any) -> str: ...

class AttrsInstance(AttrsInstance_, Protocol): ...

_A = TypeVar("_A", bound=AttrsInstance)

class _Nothing(enum.Enum):
    NOTHING = ...

NOTHING = ...
@overload
def Factory(factory: Callable[[], _T]) -> _T: ...
@overload
def Factory(factory: Callable[[Any], _T], takes_self: Literal[True]) -> _T: ...
@overload
def Factory(factory: Callable[[], _T], takes_self: Literal[False]) -> _T: ...

def __dataclass_transform__(
    *,
    eq_default: bool = ...,
    order_default: bool = ...,
    kw_only_default: bool = ...,
    frozen_default: bool = ...,
    field_descriptors: tuple[type | Callable[..., Any], ...] = ...,
) -> Callable[[_T], _T]: ...

class _CountingAttr(Generic[_T]):
    counter: int
    _default: _T
    repr: _ReprArgType
    cmp: _EqOrderType
    eq: _EqOrderType
    eq_key: str
    order: _EqOrderType
    order_key: str
    hash: bool | None
    init: bool
    converter: _ConverterType | None
    metadata: dict[Any, Any]
    _validator: _ValidatorType[_T] | None
    type: type[_T] | None
    kw_only: bool
    on_setattr: _OnSetAttrType
    alias: str | None

class Attribute(Generic[_T]):
    name: str
    default: _T | None
    validator: _ValidatorType[_T] | None
    repr: _ReprArgType
    cmp: _EqOrderType
    eq: _EqOrderType
    order: _EqOrderType
    hash: bool | None
    init: bool
    converter: _ConverterType | None
    metadata: dict[Any, Any]
    type: type[_T] | None
    kw_only: bool
    on_setattr: _OnSetAttrType
    alias: str | None
    def evolve(self, **changes: Any) -> Attribute[Any]: ...
    @classmethod
    def from_counting_attr(cls, name: str, ca: _CountingAttr[_T], type: type[Any] | None = None) -> Attribute[_T]: ...

@overload
def attrib(
    default: None = ...,
    validator: None = ...,
    repr: _ReprArgType = ...,
    cmp: _EqOrderType | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    metadata: Mapping[Any, Any] | None = ...,
    type: None = ...,
    converter: None = ...,
    factory: None = ...,
    kw_only: bool = ...,
    eq: _EqOrderType | None = ...,
    order: _EqOrderType | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    alias: str | None = ...,
) -> Any: ...
@overload
def attrib(
    default: None = ...,
    validator: _ValidatorArgType[_T] | None = ...,
    repr: _ReprArgType = ...,
    cmp: _EqOrderType | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    metadata: Mapping[Any, Any] | None = ...,
    type: type[_T] | None = ...,
    converter: _ConverterType | None = ...,
    factory: Callable[[], _T] | None = ...,
    kw_only: bool = ...,
    eq: _EqOrderType | None = ...,
    order: _EqOrderType | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    alias: str | None = ...,
) -> _T: ...
@overload
def attrib(
    default: _T,
    validator: _ValidatorArgType[_T] | None = ...,
    repr: _ReprArgType = ...,
    cmp: _EqOrderType | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    metadata: Mapping[Any, Any] | None = ...,
    type: type[_T] | None = ...,
    converter: _ConverterType | None = ...,
    factory: Callable[[], _T] | None = ...,
    kw_only: bool = ...,
    eq: _EqOrderType | None = ...,
    order: _EqOrderType | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    alias: str | None = ...,
) -> _T: ...
@overload
def attrib(
    default: _T | None = ...,
    validator: _ValidatorArgType[_T] | None = ...,
    repr: _ReprArgType = ...,
    cmp: _EqOrderType | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    metadata: Mapping[Any, Any] | None = ...,
    type: object = ...,
    converter: _ConverterType | None = ...,
    factory: Callable[[], _T] | None = ...,
    kw_only: bool = ...,
    eq: _EqOrderType | None = ...,
    order: _EqOrderType | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    alias: str | None = ...,
) -> Any: ...
@overload
def field(
    *,
    default: None = ...,
    validator: None = ...,
    repr: _ReprArgType = ...,
    hash: bool | None = ...,
    init: bool = ...,
    metadata: Mapping[Any, Any] | None = ...,
    converter: None = ...,
    factory: None = ...,
    kw_only: bool = ...,
    eq: bool | None = ...,
    order: bool | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    alias: str | None = ...,
    type: type | None = ...,
) -> Any: ...
@overload
def field(
    *,
    default: None = ...,
    validator: _ValidatorArgType[_T] | None = ...,
    repr: _ReprArgType = ...,
    hash: bool | None = ...,
    init: bool = ...,
    metadata: Mapping[Any, Any] | None = ...,
    converter: _ConverterType | None = ...,
    factory: Callable[[], _T] | None = ...,
    kw_only: bool = ...,
    eq: _EqOrderType | None = ...,
    order: _EqOrderType | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    alias: str | None = ...,
    type: type | None = ...,
) -> _T: ...
@overload
def field(
    *,
    default: _T,
    validator: _ValidatorArgType[_T] | None = ...,
    repr: _ReprArgType = ...,
    hash: bool | None = ...,
    init: bool = ...,
    metadata: Mapping[Any, Any] | None = ...,
    converter: _ConverterType | None = ...,
    factory: Callable[[], _T] | None = ...,
    kw_only: bool = ...,
    eq: _EqOrderType | None = ...,
    order: _EqOrderType | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    alias: str | None = ...,
    type: type | None = ...,
) -> _T: ...
@overload
def field(
    *,
    default: _T | None = ...,
    validator: _ValidatorArgType[_T] | None = ...,
    repr: _ReprArgType = ...,
    hash: bool | None = ...,
    init: bool = ...,
    metadata: Mapping[Any, Any] | None = ...,
    converter: _ConverterType | None = ...,
    factory: Callable[[], _T] | None = ...,
    kw_only: bool = ...,
    eq: _EqOrderType | None = ...,
    order: _EqOrderType | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    alias: str | None = ...,
    type: type | None = ...,
) -> Any: ...
@overload
@__dataclass_transform__(order_default=True, field_descriptors=(attrib, field))
def attrs(
    maybe_cls: _C,
    these: dict[str, Any] | None = ...,
    repr_ns: str | None = ...,
    repr: bool = ...,
    cmp: _EqOrderType | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: _EqOrderType | None = ...,
    order: _EqOrderType | None = ...,
    auto_detect: bool = ...,
    collect_by_mro: bool = ...,
    getstate_setstate: bool | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    field_transformer: _FieldTransformer | None = ...,
    match_args: bool = ...,
    unsafe_hash: bool | None = ...,
) -> _C: ...
@overload
@__dataclass_transform__(order_default=True, field_descriptors=(attrib, field))
def attrs(
    maybe_cls: None = ...,
    these: dict[str, Any] | None = ...,
    repr_ns: str | None = ...,
    repr: bool = ...,
    cmp: _EqOrderType | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: _EqOrderType | None = ...,
    order: _EqOrderType | None = ...,
    auto_detect: bool = ...,
    collect_by_mro: bool = ...,
    getstate_setstate: bool | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    field_transformer: _FieldTransformer | None = ...,
    match_args: bool = ...,
    unsafe_hash: bool | None = ...,
) -> Callable[[_C], _C]: ...
@overload
@__dataclass_transform__(field_descriptors=(attrib, field))
def define(
    maybe_cls: _C,
    *,
    these: dict[str, Any] | None = ...,
    repr: bool = ...,
    unsafe_hash: bool | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: bool | None = ...,
    order: bool | None = ...,
    auto_detect: bool = ...,
    getstate_setstate: bool | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    field_transformer: _FieldTransformer | None = ...,
    match_args: bool = ...,
) -> _C: ...
@overload
@__dataclass_transform__(field_descriptors=(attrib, field))
def define(
    maybe_cls: None = ...,
    *,
    these: dict[str, Any] | None = ...,
    repr: bool = ...,
    unsafe_hash: bool | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: bool | None = ...,
    order: bool | None = ...,
    auto_detect: bool = ...,
    getstate_setstate: bool | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    field_transformer: _FieldTransformer | None = ...,
    match_args: bool = ...,
) -> Callable[[_C], _C]: ...

mutable = ...

@overload
@__dataclass_transform__(frozen_default=True, field_descriptors=(attrib, field))
def frozen(
    maybe_cls: _C,
    *,
    these: dict[str, Any] | None = ...,
    repr: bool = ...,
    unsafe_hash: bool | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: bool | None = ...,
    order: bool | None = ...,
    auto_detect: bool = ...,
    getstate_setstate: bool | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    field_transformer: _FieldTransformer | None = ...,
    match_args: bool = ...,
) -> _C: ...
@overload
@__dataclass_transform__(frozen_default=True, field_descriptors=(attrib, field))
def frozen(
    maybe_cls: None = ...,
    *,
    these: dict[str, Any] | None = ...,
    repr: bool = ...,
    unsafe_hash: bool | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: bool | None = ...,
    order: bool | None = ...,
    auto_detect: bool = ...,
    getstate_setstate: bool | None = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    field_transformer: _FieldTransformer | None = ...,
    match_args: bool = ...,
) -> Callable[[_C], _C]: ...
def fields(cls: type[AttrsInstance]) -> Any: ...
def fields_dict(cls: type[AttrsInstance]) -> dict[str, Attribute[Any]]: ...
def validate(inst: AttrsInstance) -> None: ...
def resolve_types(
    cls: _A,
    globalns: dict[str, Any] | None = ...,
    localns: dict[str, Any] | None = ...,
    attribs: list[Attribute[Any]] | None = ...,
    include_extras: bool = ...,
) -> _A: ...
def make_class(
    name: str,
    attrs: list[str] | tuple[str, ...] | dict[str, Any],
    bases: tuple[type, ...] = ...,
    repr_ns: str | None = ...,
    repr: bool = ...,
    cmp: _EqOrderType | None = ...,
    hash: bool | None = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: _EqOrderType | None = ...,
    order: _EqOrderType | None = ...,
    collect_by_mro: bool = ...,
    on_setattr: _OnSetAttrArgType | None = ...,
    field_transformer: _FieldTransformer | None = ...,
) -> type: ...
def asdict(
    inst: AttrsInstance,
    recurse: bool = ...,
    filter: _FilterType[Any] | None = ...,
    dict_factory: type[Mapping[Any, Any]] = ...,
    retain_collection_types: bool = ...,
    value_serializer: Callable[[type, Attribute[Any], Any], Any] | None = ...,
    tuple_keys: bool | None = ...,
) -> dict[str, Any]: ...
def astuple(
    inst: AttrsInstance,
    recurse: bool = ...,
    filter: _FilterType[Any] | None = ...,
    tuple_factory: type[Sequence[Any]] = ...,
    retain_collection_types: bool = ...,
) -> tuple[Any, ...]: ...
def has(cls: type) -> TypeGuard[type[AttrsInstance]]: ...
def assoc(inst: _T, **changes: Any) -> _T: ...
def evolve(inst: _T, **changes: Any) -> _T: ...
def set_run_validators(run: bool) -> None: ...
def get_run_validators() -> bool: ...

attributes = ...
attr = ...
dataclass = ...

def _make_init(
    cls: type[AttrsInstance],
    attrs: tuple[Attribute[Any]],
    pre_init: bool,
    post_init: bool,
    frozen: bool,
    slots: bool,
    cache_hash: bool,
    base_attr_map: dict[str, Any],
    is_exc: bool,
    cls_on_setattr: Any,
    attrs_init: bool,
) -> Callable[_P, Any]: ...
def _make_method(name: str, script: str, filename: str, globs: dict[str, Any]) -> Callable[..., Any]: ...
def _make_repr(attrs: tuple[Attribute[Any]], ns: str | None, cls: AttrsInstance) -> ReprProtocol: ...
def _transform_attrs(
    cls: type[AttrsInstance],
    these: dict[str, _CountingAttr[_T]] | None,
    auto_attribs: bool,
    kw_only: bool,
    collect_by_mro: bool,
    field_transformer: _FieldTransformer | None,
) -> tuple[tuple[attr.Attribute[Any], ...], tuple[attr.Attribute[Any], ...], dict[attr.Attribute[Any], type[Any]]]: ...
