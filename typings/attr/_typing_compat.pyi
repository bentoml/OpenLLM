from typing import Any, ClassVar, Protocol

MYPY = False

if MYPY:
    class AttrsInstance_(Protocol):
        __attrs_attrs__: ClassVar[Any]
        ...

else:
    class AttrsInstance_(Protocol): ...
