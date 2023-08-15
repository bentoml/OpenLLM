from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Type, TypeVar, Union, overload

import click

from ._core import _FC, AnyCallable, OptionGroup

class OptionStackItem(NamedTuple):
    param_decls: Tuple[str, ...]
    attrs: Dict[str, Any]
    param_count: int

class _NotAttachedOption(click.Option):
    def __init__(self, param_decls: Any = ..., *, all_not_attached_options: Any, **attrs: Any) -> None: ...
    def handle_parse_result(self, ctx: click.Context, opts: Any, args: List[str]) -> Any: ...

_GrpType = TypeVar("_GrpType", bound=OptionGroup)

class _OptGroup:
    def __init__(self) -> None: ...
    def __call__(
        self,
        name: str,
        *,
        help: Optional[str] = None,
        cls: Optional[Type[_GrpType]] = None,
        **attrs: Any,
    ) -> Union[click.Command, Callable[[AnyCallable], click.Command]]: ...
    @overload
    def group(
        self,
        name: str,
        *,
        cls: type[_GrpType],
        **attrs: Any,
    ) -> Callable[[AnyCallable], click.Command]: ...
    @overload
    def group(
        self,
        name: str,
        cls: None,
        **attrs: Any,
    ) -> Callable[[AnyCallable], click.Command]: ...
    @overload
    def group(
        self,
        *,
        name: str,
        **attrs: Any,
    ) -> Callable[[AnyCallable], click.Command]: ...
    @overload
    def group(
        self,
        name: Optional[str] = ...,
        *,
        help: Optional[str] = ...,
        cls: Optional[Type[_GrpType]] = None,
        **attrs: Any,
    ) -> Union[click.Command, Callable[[AnyCallable], click.Command]]: ...
    def option(self, *param_decls: Any, **attrs: Any) -> Callable[[_FC], _FC]: ...

optgroup: _OptGroup = ...
