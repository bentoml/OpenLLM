from typing import Any
from typing import Callable
from typing import Optional
from typing import TypeAlias

_CompareWithType: TypeAlias = Callable[[Any, Any], bool]

def cmp_using(
    eq: Optional[_CompareWithType] = ...,
    lt: Optional[_CompareWithType] = ...,
    le: Optional[_CompareWithType] = ...,
    gt: Optional[_CompareWithType] = ...,
    ge: Optional[_CompareWithType] = ...,
    require_same_type: bool = ...,
    class_name: str = ...,
) -> type[Any]: ...
