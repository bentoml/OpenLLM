from typing import Any
from typing import ClassVar
from typing import Protocol


MYPY = False

if MYPY:
    class AttrsInstance_(Protocol):
        __attrs_attrs__: ClassVar[Any]
        ...

else:
    class AttrsInstance_(Protocol): ...
