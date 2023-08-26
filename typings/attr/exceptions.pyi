from typing import Any
class FrozenError(AttributeError):
    msg: str = ...

class FrozenInstanceError(FrozenError): ...
class FrozenAttributeError(FrozenError): ...
class AttrsAttributeNotFoundError(ValueError): ...
class NotAnAttrsClassError(ValueError): ...
class DefaultAlreadySetError(RuntimeError): ...
class UnannotatedAttributeError(RuntimeError): ...
class PythonTooOldError(RuntimeError): ...

class NotCallableError(TypeError):
    msg: str = ...
    value: Any = ...
    def __init__(self, msg: str, value: Any) -> None: ...
