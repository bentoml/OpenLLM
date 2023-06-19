from ._core import AllOptionGroup
from ._core import GroupedOption
from ._core import MutuallyExclusiveOptionGroup
from ._core import OptionGroup
from ._core import RequiredAllOptionGroup
from ._core import RequiredAnyOptionGroup
from ._core import RequiredMutuallyExclusiveOptionGroup
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
