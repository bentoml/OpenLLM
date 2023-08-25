from typing import Any, Callable, Generator, Generic, Tuple, TypeVar, Union, overload

from _typeshed import Incomplete

class _SentinelClass: ...
_VT = TypeVar('_VT')

class Provider(Generic[_VT]):
    STATE_FIELDS: Tuple[str, ...]
    def __init__(self) -> None: ...
    def set(self, value: Union[_SentinelClass, _VT]) -> None: ...
    def patch(self, value: Union[_SentinelClass, _VT]) -> Generator[None, None, None]: ...
    def get(self) -> _VT: ...
    def reset(self) -> None: ...

class _ProvideClass:
    def __getitem__(self, provider: Provider[_VT]) -> _VT: ...

Provide: Incomplete
_AnyCallable = TypeVar('_AnyCallable', bound=Callable[..., Any])


@overload
def inject(func: _AnyCallable) -> _AnyCallable: ...
@overload
def inject(func: None = ..., squeeze_none: bool = ...) -> Callable[[_AnyCallable], _AnyCallable]: ...
def sync_container(from_: Any, to_: Any) -> None: ...

container: Incomplete
