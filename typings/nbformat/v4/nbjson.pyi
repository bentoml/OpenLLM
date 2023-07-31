import json
from typing import Any
from typing import Dict

from _typeshed import Incomplete
from nbformat import NotebookNode
from nbformat.notebooknode import from_dict as from_dict

from .rwbase import NotebookReader as NotebookReader
from .rwbase import NotebookWriter as NotebookWriter
from .rwbase import rejoin_lines as rejoin_lines
from .rwbase import split_lines as split_lines
from .rwbase import strip_transient as strip_transient

class BytesEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any: ...

class JSONReader(NotebookReader):
    def reads(self, s: str, **kwargs: Any) -> NotebookNode: ...
    def to_notebook(self, d: Dict[str, Any], **kwargs: Any) -> NotebookNode: ...

class JSONWriter(NotebookWriter):
    def writes(self, nb: NotebookNode, **kwargs: Any) -> None: ...

reads: Incomplete
read: Incomplete
to_notebook: Incomplete
write: Incomplete
writes: Incomplete
