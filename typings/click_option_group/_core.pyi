from collections.abc import Callable
from typing import Any
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union

import click


FC = Union[Callable[..., Any], click.Command]

class GroupedOption(click.Option):
    """Represents grouped (related) optional values

    The class should be used only with `OptionGroup` class for creating grouped options.

    :param param_decls: option declaration tuple
    :param group: `OptionGroup` instance (the group for this option)
    :param attrs: additional option attributes
    """

    def __init__(self, param_decls: Optional[Sequence[str]] = ..., *, group: OptionGroup, **attrs: Any) -> None: ...
    @property
    def group(self) -> OptionGroup:
        """Returns the reference to the group for this option

        :return: `OptionGroup` the group instance for this option
        """
        ...
    def handle_parse_result(
        self, ctx: click.Context, opts: Mapping[str, Any], args: List[str]
    ) -> Tuple[Any, List[str]]: ...
    def get_help_record(self, ctx: click.Context) -> Optional[Tuple[str, str]]: ...

class _GroupTitleFakeOption(click.Option):
    """The helper `Option` class to display option group title in help"""

    def __init__(self, param_decls: Optional[Sequence[str]] = ..., *, group: OptionGroup, **attrs: Any) -> None: ...
    def get_help_record(self, ctx: click.Context) -> Optional[Tuple[str, str]]: ...

class OptionGroup:
    """Option group manages grouped (related) options

    The class is used for creating the groups of options. The class can de used as based class to implement
    specific behavior for grouped options.

    :param name: the group name. If it is not set the default group name will be used
    :param help: the group help text or None
    """

    def __init__(self, name: Optional[str] = ..., *, hidden: bool = ..., help: Optional[str] = ...) -> None: ...
    @property
    def name(self) -> str:
        """Returns the group name or empty string if it was not set

        :return: group name
        """
        ...
    @property
    def help(self) -> str:
        """Returns the group help or empty string if it was not set

        :return: group help
        """
        ...
    @property
    def name_extra(self) -> List[str]:
        """Returns extra name attributes for the group"""
        ...
    @property
    def forbidden_option_attrs(self) -> List[str]:
        """Returns the list of forbidden option attributes for the group"""
        ...
    def get_help_record(self, ctx: click.Context) -> Optional[Tuple[str, str]]:
        """Returns the help record for the group

        :param ctx: Click Context object
        :return: the tuple of two fileds: `(name, help)`
        """
        ...
    def option(self, *param_decls: str, **attrs: Any) -> Callable:
        """Decorator attaches an grouped option to the command

        The decorator is used for adding options to the group and to the Click-command
        """
        ...
    def get_options(self, ctx: click.Context) -> Dict[str, GroupedOption]:
        """Returns the dictionary with group options"""
        ...
    def get_option_names(self, ctx: click.Context) -> List[str]:
        """Returns the list with option names ordered by addition in the group"""
        ...
    def get_error_hint(self, ctx: click.Context, option_names: Optional[Set[str]] = ...) -> str: ...
    def handle_parse_result(self, option: GroupedOption, ctx: click.Context, opts: Mapping[str, Any]) -> None:
        """The method should be used for adding specific behavior and relation for options in the group"""
        ...

class RequiredAnyOptionGroup(OptionGroup):
    """Option group with required any options of this group

    `RequiredAnyOptionGroup` defines the behavior: At least one option from the group must be set.
    """

    @property
    def forbidden_option_attrs(self) -> List[str]: ...
    @property
    def name_extra(self) -> List[str]: ...
    def handle_parse_result(self, option: GroupedOption, ctx: click.Context, opts: Mapping[str, Any]) -> None: ...

class RequiredAllOptionGroup(OptionGroup):
    """Option group with required all options of this group

    `RequiredAllOptionGroup` defines the behavior: All options from the group must be set.
    """

    @property
    def forbidden_option_attrs(self) -> List[str]: ...
    @property
    def name_extra(self) -> List[str]: ...
    def handle_parse_result(self, option: GroupedOption, ctx: click.Context, opts: Mapping[str, Any]) -> None: ...

class MutuallyExclusiveOptionGroup(OptionGroup):
    """Option group with mutually exclusive behavior for grouped options

    `MutuallyExclusiveOptionGroup` defines the behavior:
        - Only one or none option from the group must be set
    """

    @property
    def forbidden_option_attrs(self) -> List[str]: ...
    @property
    def name_extra(self) -> List[str]: ...
    def handle_parse_result(self, option: GroupedOption, ctx: click.Context, opts: Mapping[str, Any]) -> None: ...

class RequiredMutuallyExclusiveOptionGroup(MutuallyExclusiveOptionGroup):
    """Option group with required and mutually exclusive behavior for grouped options

    `RequiredMutuallyExclusiveOptionGroup` defines the behavior:
        - Only one required option from the group must be set
    """

    @property
    def name_extra(self) -> List[str]: ...
    def handle_parse_result(self, option: GroupedOption, ctx: click.Context, opts: Mapping[str, Any]) -> None: ...

class AllOptionGroup(OptionGroup):
    """Option group with required all/none options of this group

    `AllOptionGroup` defines the behavior:
        - All options from the group must be set or None must be set
    """

    @property
    def forbidden_option_attrs(self) -> List[str]: ...
    @property
    def name_extra(self) -> List[str]: ...
    def handle_parse_result(self, option: GroupedOption, ctx: click.Context, opts: Mapping[str, Any]) -> None: ...
