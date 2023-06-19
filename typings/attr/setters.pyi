from typing import Any
from typing import NewType
from typing import NoReturn
from typing import TypeVar

from . import Attribute
from . import _OnSetAttrType


_T = TypeVar("_T")

def frozen(instance: Any, attribute: Attribute[Any], new_value: Any) -> NoReturn: ...
def pipe(*setters: _OnSetAttrType) -> _OnSetAttrType: ...
def validate(instance: Any, attribute: Attribute[_T], new_value: _T) -> _T: ...
def convert(instance: Any, attribute: Attribute[Any], new_value: Any) -> Any: ...

_NoOpType = NewType("_NoOpType", object)
NO_OP: _NoOpType
