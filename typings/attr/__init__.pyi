import enum
import sys
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    ParamSpec,
    Protocol,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    overload,
)

if sys.version_info[:2] >= (3, 11):
  from typing import dataclass_transform
else:
  from typing_extensions import dataclass_transform

from . import (
    converters as converters,
    exceptions as exceptions,
    filters as filters,
    setters as setters,
    validators as validators,
)
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
_EqOrderType: TypeAlias = Union[bool, Callable[[Any], Any]]
_ValidatorType: TypeAlias = Callable[[Any, Attribute[_T], _T], Any]
_ConverterType: TypeAlias = Callable[[Any], Any]
_FilterType: TypeAlias = Callable[[Attribute[_T], _T], bool]
_ReprType: TypeAlias = Callable[[Any], str]
_ReprArgType: TypeAlias = Union[bool, _ReprType]
_OnSetAttrType: TypeAlias = Callable[[Any, Attribute[Any], Any], Any]
_OnSetAttrArgType: TypeAlias = Union[_OnSetAttrType, List[_OnSetAttrType], setters._NoOpType]
_FieldTransformer: TypeAlias = Callable[[type, List[Attribute[Any]]], List[Attribute[Any]]]
_ValidatorArgType: TypeAlias = Union[_ValidatorType[_T], Sequence[_ValidatorType[_T]]]

class AttrsInstance(AttrsInstance_, Protocol): ...

_A = TypeVar("_A", bound=AttrsInstance)

class _Nothing(enum.Enum):
    NOTHING = ...

NOTHING: enum.Enum = ...

@overload
def Factory(factory: Callable[[], _T]) -> _T: ...
@overload
def Factory(factory: Callable[[Any], _T], takes_self: Literal[True]) -> _T: ...
@overload
def Factory(factory: Callable[[], _T], takes_self: Literal[False]) -> _T: ...

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
    def from_counting_attr(cls, name: str, ca: _CountingAttr[_T], type: Type[Any] | None = None) -> Attribute[_T]: ...

# NOTE: We had several choices for the annotation to use for type arg:
# 1) Type[_T]
#   - Pros: Handles simple cases correctly
#   - Cons: Might produce less informative errors in the case of conflicting
#     TypeVars e.g. `attr.ib(default='bad', type=int)`
# 2) Callable[..., _T]
#   - Pros: Better error messages than #1 for conflicting TypeVars
#   - Cons: Terrible error messages for validator checks.
#   e.g. attr.ib(type=int, validator=validate_str)
#        -> error: Cannot infer function type argument
# 3) type (and do all of the work in the mypy plugin)
#   - Pros: Simple here, and we could customize the plugin with our own errors.
#   - Cons: Would need to write mypy plugin code to handle all the cases.
# We chose option #1.

# `attr` lies about its return type to make the following possible:
#     attr()    -> Any
#     attr(8)   -> int
#     attr(validator=<some callable>)  -> Whatever the callable expects.
# This makes this type of assignments possible:
#     x: int = attr(8)
#
# This form catches explicit None or no default but with no other arguments
# returns Any.
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

# This form catches an explicit None or no default and infers the type from the
# other arguments.
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

# This form catches an explicit default argument.
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

# This form covers type=non-Type: e.g. forward references (str), Any
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

# This form catches an explicit None or no default and infers the type from the
# other arguments.
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

# This form catches an explicit default argument.
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

# This form covers type=non-Type: e.g. forward references (str), Any
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
@dataclass_transform(order_default=True, field_specifiers=(attrib, field))
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
@dataclass_transform(order_default=True, field_specifiers=(attrib, field))
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
@dataclass_transform(field_specifiers=(attrib, field))
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
@dataclass_transform(field_specifiers=(attrib, field))
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

mutable = define

@overload
@dataclass_transform(frozen_default=True, field_specifiers=(attrib, field))
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
@dataclass_transform(frozen_default=True, field_specifiers=(attrib, field))
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
@overload
def resolve_types(
    cls: Type[_A],
    globalns: Optional[Dict[str, Any]] = ...,
    localns: Optional[Dict[str, Any]] = ...,
    attribs: Optional[List[Attribute[Any]]] = ...,
    include_extras: bool = ...,
) -> Type[_A]: ...
@overload
def resolve_types(
    cls: _A,
    globalns: Optional[Dict[str, Any]] = ...,
    localns: Optional[Dict[str, Any]] = ...,
    attribs: Optional[List[Attribute[Any]]] = ...,
    include_extras: bool = ...,
) -> _A: ...

# TODO: add support for returning a proper attrs class from the mypy plugin
# we use Any instead of _CountingAttr so that e.g. `make_class('Foo',
# [attr.ib()])` is valid
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

# _funcs --

# TODO: add support for returning TypedDict from the mypy plugin
# FIXME: asdict/astuple do not honor their factory args. Waiting on one of
# these:
# https://github.com/python/mypy/issues/4236
# https://github.com/python/typing/issues/253
# XXX: remember to fix attrs.asdict/astuple too!
def asdict(
    inst: AttrsInstance,
    recurse: bool = ...,
    filter: Optional[_FilterType[Any]] = ...,
    dict_factory: Type[Mapping[Any, Any]] = ...,
    retain_collection_types: bool = ...,
    value_serializer: Optional[Callable[[type, Attribute[Any], Any], Any]] = ...,
    tuple_keys: Optional[bool] = ...,
) -> Dict[str, Any]: ...

# TODO: add support for returning NamedTuple from the mypy plugin
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

# _config --
def set_run_validators(run: bool) -> None: ...
def get_run_validators() -> bool: ...

# aliases --
s = attrs
attributes = attrs
ib = attrib
attr = attrib
dataclass = attrs  # Technically, partial(attrs, auto_attribs=True) ;)

class ReprProtocol(Protocol):
  def __call__(__self, self: Any) -> str: ...
def _make_init(
    cls: type[AttrsInstance],
    attrs: tuple[Attribute[Any], ...],
    pre_init: bool,
    post_init: bool,
    frozen: bool,
    slots: bool,
    cache_hash: bool,
    base_attr_map: dict[Any, Any],
    is_exc: bool,
    cls_on_setattr: Any,
    attrs_init: bool,
) -> Callable[_P, Any]: ...
def _make_repr(attrs: tuple[Attribute[Any]], ns: str | None, cls: AttrsInstance) -> ReprProtocol: ...
def _transform_attrs(
    cls: type[AttrsInstance],
    these: dict[str, _CountingAttr[_T]] | None,
    auto_attribs: bool,
    kw_only: bool,
    collect_by_mro: bool,
    field_transformer: _FieldTransformer | None,
) -> tuple[tuple[Attribute[_T], ...], tuple[Attribute[_T], ...], dict[Attribute[_T], type[Any]]]: ...
