from ._core import (
    AllOptionGroup,
    GroupedOption,
    MutuallyExclusiveOptionGroup,
    OptionGroup,
    RequiredAllOptionGroup,
    RequiredAnyOptionGroup,
    RequiredMutuallyExclusiveOptionGroup,
)
from ._decorators import optgroup
from ._version import __version__

"""
click-option-group
~~~~~~~~~~~~~~~~~~

Option groups missing in Click

:copyright: © 2019-2020 by Eugene Prilepin
:license: BSD, see LICENSE for more details.
"""
__all__ = [
    "__version__",
    "optgroup",
    "GroupedOption",
    "OptionGroup",
    "RequiredAnyOptionGroup",
    "AllOptionGroup",
    "RequiredAllOptionGroup",
    "MutuallyExclusiveOptionGroup",
    "RequiredMutuallyExclusiveOptionGroup",
]
