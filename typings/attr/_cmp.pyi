import sys
from typing import Any, Callable, Optional
if sys.version_info[:2] >= (3, 10):
  from typing import TypeAlias
else:
  from typing_extensions import TypeAlias

_CompareWithType: TypeAlias = Callable[[Any, Any], bool]

def cmp_using(eq: Optional[_CompareWithType] = ..., lt: Optional[_CompareWithType] = ..., le: Optional[_CompareWithType] = ..., gt: Optional[_CompareWithType] = ..., ge: Optional[_CompareWithType] = ..., require_same_type: bool = ..., class_name: str = ...,) -> type[Any]: ...
