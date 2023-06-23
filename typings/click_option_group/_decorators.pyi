from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import NamedTuple
from typing import Optional
from typing import ParamSpec
from typing import Protocol
from typing import Tuple
from typing import Type
from typing import TypeVar

import click

from ._core import FC
from ._core import OptionGroup


P = ParamSpec("P")
O_co = TypeVar("O_co", covariant=True)

F = Callable[P, O_co]

class OptionStackItem(NamedTuple):
    param_decls: Tuple[str, ...]
    attrs: Dict[str, Any]
    param_count: int
    ...

class ClickFunctionWrapper(Protocol[P, O_co]):
    __name__: str
    __click_params__: list[click.Option]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> O_co: ...

class _NotAttachedOption(click.Option):
    """The helper class to catch grouped options which were not attached to the group

    Raises TypeError if not attached options exist.
    """

    def __init__(self, param_decls: Any = ..., *, all_not_attached_options: Any, **attrs: Any) -> None: ...
    def handle_parse_result(self, ctx: click.Context, opts: Any, args: tuple[Any]) -> Any: ...

class _OptGroup(Generic[O_co]):
    """A helper class to manage creating groups and group options via decorators

    The class provides two decorator-methods: `group`/`__call__` and `option`.
    These decorators should be used for adding grouped options. The class have
    single global instance `optgroup` that should be used in most cases.

    The example of usage::

        ...
        @optgroup('Group 1', help='option group 1')
        @optgroup.option('--foo')
        @optgroup.option('--bar')
        @optgroup.group('Group 2', help='option group 2')
        @optgroup.option('--spam')
        ...
    """

    def __init__(self) -> None: ...
    def __call__(
        self,
        name: Optional[str] = ...,
        *,
        help: Optional[str] = ...,
        cls: Optional[Type[OptionGroup]] = ...,
        **attrs: Any,
    ) -> FC:
        """Creates a new group and collects its options

        Creates the option group and registers all grouped options
        which were added by `option` decorator.

        :param name: Group name or None for default name
        :param help: Group help or None for empty help
        :param cls: Option group class that should be inherited from `OptionGroup` class
        :param attrs: Additional parameters of option group class
        """
        ...
    def group(
        self,
        name: Optional[str] = ...,
        *,
        help: Optional[str] = ...,
        cls: Optional[Type[OptionGroup]] = ...,
        **attrs: Any,
    ) -> FC:
        """The decorator creates a new group and collects its options

        Creates the option group and registers all grouped options
        which were added by `option` decorator.

        :param name: Group name or None for default name
        :param help: Group help or None for empty help
        :param cls: Option group class that should be inherited from `OptionGroup` class
        :param attrs: Additional parameters of option group class
        """
        ...
    def option(self, *param_decls: Any, **attrs: Any) -> FC:
        """The decorator adds a new option to the group

        The decorator is lazy. It adds option decls and attrs.
        All options will be registered by `group` decorator.

        :param param_decls: option declaration tuple
        :param attrs: additional option attributes and parameters
        """
        ...

optgroup: _OptGroup[Any] = ...
