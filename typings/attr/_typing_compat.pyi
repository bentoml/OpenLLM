from typing import Any
from typing import ClassVar
from typing import Protocol

class AttrsInstance_(Protocol):
    __attrs_attrs__: ClassVar[Any]
    def __attrs_init__(self, *args: Any, **kwargs: Any) -> None: ...
