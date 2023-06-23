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


if sys.version_info >= (3, 10): ...
else: ...
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
if sys.version_info >= (3, 8):
    @overload
    def Factory(factory: Callable[[], _T]) -> _T: ...
    @overload
    def Factory(factory: Callable[[Any], _T], takes_self: Literal[True]) -> _T: ...
    @overload
    def Factory(factory: Callable[[], _T], takes_self: Literal[False]) -> _T: ...

else: ...

def __dataclass_transform__(
    *,
    eq_default: bool = ...,
    order_default: bool = ...,
    kw_only_default: bool = ...,
    frozen_default: bool = ...,
    field_descriptors: Tuple[Union[type, Callable[..., Any]], ...] = ...,
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
    hash: Optional[bool]
    init: bool
    converter: Optional[_ConverterType]
    metadata: Dict[Any, Any]
    _validator: Optional[_ValidatorType[_T]]
    type: Optional[Type[_T]]
    kw_only: bool
    on_setattr: _OnSetAttrType
    alias: Optional[str]

class Attribute(Generic[_T]):
    name: str
    default: Optional[_T]
    validator: Optional[_ValidatorType[_T]]
    repr: _ReprArgType
    cmp: _EqOrderType
    eq: _EqOrderType
    order: _EqOrderType
    hash: Optional[bool]
    init: bool
    converter: Optional[_ConverterType]
    metadata: Dict[Any, Any]
    type: Optional[Type[_T]]
    kw_only: bool
    on_setattr: _OnSetAttrType
    alias: Optional[str]
    def evolve(self, **changes: Any) -> Attribute[Any]: ...
    @classmethod
    def from_counting_attr(cls, name: str, ca: _CountingAttr[_T], type: type[Any] | None = None) -> Attribute[_T]: ...

@overload
def attrib(
    default: None = ...,
    validator: None = ...,
    repr: _ReprArgType = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    type: None = ...,
    converter: None = ...,
    factory: None = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    alias: Optional[str] = ...,
) -> Any: ...
@overload
def attrib(
    default: None = ...,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    type: Optional[Type[_T]] = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    alias: Optional[str] = ...,
) -> _T: ...
@overload
def attrib(
    default: _T,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    type: Optional[Type[_T]] = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    alias: Optional[str] = ...,
) -> _T: ...
@overload
def attrib(
    default: Optional[_T] = ...,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    type: object = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    alias: Optional[str] = ...,
) -> Any: ...
@overload
def field(
    *,
    default: None = ...,
    validator: None = ...,
    repr: _ReprArgType = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    converter: None = ...,
    factory: None = ...,
    kw_only: bool = ...,
    eq: Optional[bool] = ...,
    order: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    alias: Optional[str] = ...,
    type: Optional[type] = ...,
) -> Any: ...
@overload
def field(
    *,
    default: None = ...,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    alias: Optional[str] = ...,
    type: Optional[type] = ...,
) -> _T: ...
@overload
def field(
    *,
    default: _T,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    alias: Optional[str] = ...,
    type: Optional[type] = ...,
) -> _T: ...
@overload
def field(
    *,
    default: Optional[_T] = ...,
    validator: Optional[_ValidatorArgType[_T]] = ...,
    repr: _ReprArgType = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    metadata: Optional[Mapping[Any, Any]] = ...,
    converter: Optional[_ConverterType] = ...,
    factory: Optional[Callable[[], _T]] = ...,
    kw_only: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    alias: Optional[str] = ...,
    type: Optional[type] = ...,
) -> Any: ...
@overload
@__dataclass_transform__(order_default=True, field_descriptors=(attrib, field))
def attrs(
    maybe_cls: _C,
    these: Optional[Dict[str, Any]] = ...,
    repr_ns: Optional[str] = ...,
    repr: bool = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    auto_detect: bool = ...,
    collect_by_mro: bool = ...,
    getstate_setstate: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
    match_args: bool = ...,
    unsafe_hash: Optional[bool] = ...,
) -> _C: ...
@overload
@__dataclass_transform__(order_default=True, field_descriptors=(attrib, field))
def attrs(
    maybe_cls: None = ...,
    these: Optional[Dict[str, Any]] = ...,
    repr_ns: Optional[str] = ...,
    repr: bool = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    auto_detect: bool = ...,
    collect_by_mro: bool = ...,
    getstate_setstate: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
    match_args: bool = ...,
    unsafe_hash: Optional[bool] = ...,
) -> Callable[[_C], _C]: ...
@overload
@__dataclass_transform__(field_descriptors=(attrib, field))
def define(
    maybe_cls: _C,
    *,
    these: Optional[Dict[str, Any]] = ...,
    repr: bool = ...,
    unsafe_hash: Optional[bool] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[bool] = ...,
    order: Optional[bool] = ...,
    auto_detect: bool = ...,
    getstate_setstate: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
    match_args: bool = ...,
) -> _C: ...
@overload
@__dataclass_transform__(field_descriptors=(attrib, field))
def define(
    maybe_cls: None = ...,
    *,
    these: Optional[Dict[str, Any]] = ...,
    repr: bool = ...,
    unsafe_hash: Optional[bool] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[bool] = ...,
    order: Optional[bool] = ...,
    auto_detect: bool = ...,
    getstate_setstate: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
    match_args: bool = ...,
) -> Callable[[_C], _C]: ...

mutable = ...

@overload
@__dataclass_transform__(frozen_default=True, field_descriptors=(attrib, field))
def frozen(
    maybe_cls: _C,
    *,
    these: Optional[Dict[str, Any]] = ...,
    repr: bool = ...,
    unsafe_hash: Optional[bool] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[bool] = ...,
    order: Optional[bool] = ...,
    auto_detect: bool = ...,
    getstate_setstate: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
    match_args: bool = ...,
) -> _C: ...
@overload
@__dataclass_transform__(frozen_default=True, field_descriptors=(attrib, field))
def frozen(
    maybe_cls: None = ...,
    *,
    these: Optional[Dict[str, Any]] = ...,
    repr: bool = ...,
    unsafe_hash: Optional[bool] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[bool] = ...,
    order: Optional[bool] = ...,
    auto_detect: bool = ...,
    getstate_setstate: Optional[bool] = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
    match_args: bool = ...,
) -> Callable[[_C], _C]: ...
def fields(cls: Type[AttrsInstance]) -> Any: ...
def fields_dict(cls: Type[AttrsInstance]) -> Dict[str, Attribute[Any]]: ...
def validate(inst: AttrsInstance) -> None: ...
def resolve_types(
    cls: _A,
    globalns: Optional[Dict[str, Any]] = ...,
    localns: Optional[Dict[str, Any]] = ...,
    attribs: Optional[List[Attribute[Any]]] = ...,
    include_extras: bool = ...,
) -> _A: ...
def make_class(
    name: str,
    attrs: Union[List[str], Tuple[str, ...], Dict[str, Any]],
    bases: Tuple[type, ...] = ...,
    repr_ns: Optional[str] = ...,
    repr: bool = ...,
    cmp: Optional[_EqOrderType] = ...,
    hash: Optional[bool] = ...,
    init: bool = ...,
    slots: bool = ...,
    frozen: bool = ...,
    weakref_slot: bool = ...,
    str: bool = ...,
    auto_attribs: bool = ...,
    kw_only: bool = ...,
    cache_hash: bool = ...,
    auto_exc: bool = ...,
    eq: Optional[_EqOrderType] = ...,
    order: Optional[_EqOrderType] = ...,
    collect_by_mro: bool = ...,
    on_setattr: Optional[_OnSetAttrArgType] = ...,
    field_transformer: Optional[_FieldTransformer] = ...,
) -> type: ...
def asdict(
    inst: AttrsInstance,
    recurse: bool = ...,
    filter: Optional[_FilterType[Any]] = ...,
    dict_factory: Type[Mapping[Any, Any]] = ...,
    retain_collection_types: bool = ...,
    value_serializer: Optional[Callable[[type, Attribute[Any], Any], Any]] = ...,
    tuple_keys: Optional[bool] = ...,
) -> Dict[str, Any]: ...
def astuple(
    inst: AttrsInstance,
    recurse: bool = ...,
    filter: Optional[_FilterType[Any]] = ...,
    tuple_factory: Type[Sequence[Any]] = ...,
    retain_collection_types: bool = ...,
) -> Tuple[Any, ...]: ...
def has(cls: type) -> TypeGuard[Type[AttrsInstance]]: ...
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
    field_transformer: Optional[_FieldTransformer],
) -> tuple[tuple[attr.Attribute[Any], ...], tuple[attr.Attribute[Any], ...], dict[attr.Attribute[Any], type[Any]]]: ...
