from typing import Any

from ._struct import Struct

class NotebookNode(Struct): ...

def from_dict(d: dict[str, Any]) -> NotebookNode: ...
