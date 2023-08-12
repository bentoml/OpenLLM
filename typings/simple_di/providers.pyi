import sys
from typing import (
    Any,
    Callable as CallableType,
    Dict,
    Tuple,
    Union,
)
if sys.version_info[:2] >= (3, 10):
  from typing import TypeAlias
else:
  from typing_extensions import TypeAlias


from _typeshed import Incomplete

from . import _VT, Provider, _SentinelClass

class Placeholder(Provider[_VT]): ...

class Static(Provider[_VT]):
    STATE_FIELDS: Tuple[str, ...]
    def __init__(self, value: _VT) -> None: ...

class Factory(Provider[_VT]):
    STATE_FIELDS: Tuple[str, ...]
    def __init__(self, func: CallableType[..., _VT], *args: Any, **kwargs: Any) -> None: ...

class SingletonFactory(Factory[_VT]):
    STATE_FIELDS: Tuple[str, ...]
    def __init__(self, func: CallableType[..., _VT], *args: Any, **kwargs: Any) -> None: ...
Callable = Factory
MemoizedCallable = SingletonFactory
ConfigDictType: TypeAlias = Dict[Union[str, int], Any]
PathItemType: TypeAlias = Union[int, str, Provider[int], Provider[str]]

class Configuration(Provider[ConfigDictType]):
    STATE_FIELDS: Tuple[str, ...]
    fallback: Incomplete
    def __init__(self, data: Union[_SentinelClass, ConfigDictType] = ..., fallback: Any = ...) -> None: ...
    def set(self, value: Union[_SentinelClass, ConfigDictType]) -> None: ...
    def get(self) -> Union[ConfigDictType, Any]: ...
    def reset(self) -> None: ...
    def __getattr__(self, name: str) -> _ConfigurationItem: ...
    def __getitem__(self, key: PathItemType) -> _ConfigurationItem: ...

class _ConfigurationItem(Provider[Any]):
    STATE_FIELDS: Tuple[str, ...]
    def __init__(self, config: Configuration, path: Tuple[PathItemType, ...]) -> None: ...
    def set(self, value: Any) -> None: ...
    def get(self) -> Any: ...
    def reset(self) -> None: ...
    def __getattr__(self, name: str) -> _ConfigurationItem: ...
    def __getitem__(self, key: PathItemType) -> _ConfigurationItem: ...
