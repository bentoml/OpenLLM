from typing import Any
from typing import Union

from . import Attribute
from . import _FilterType


def include(*what: Union[type, str, Attribute[Any]]) -> _FilterType[Any]: ...
def exclude(*what: Union[type, str, Attribute[Any]]) -> _FilterType[Any]: ...
