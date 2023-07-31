from typing import IO
from typing import Any

from _typeshed import Incomplete
from nbformat import NotebookNode
from nbformat.v4.rwbase import NotebookReader
from nbformat.v4.rwbase import NotebookWriter

from .config import JupytextConfiguration

class NotSupportedNBFormatVersion(NotImplementedError): ...

class TextNotebookConverter(NotebookReader, NotebookWriter):
    fmt: Incomplete
    config: Incomplete
    ext: Incomplete
    implementation: Incomplete
    def __init__(self, fmt: Any, config: JupytextConfiguration) -> None: ...
    def update_fmt_with_notebook_options(self, metadata: Any, read: bool = ...) -> None: ...
    def reads(self, s: str, **_: Any) -> NotebookNode: ...
    def filter_notebook(self, nb: NotebookNode, metadata: Any) -> Any: ...
    def writes(self, nb: NotebookNode, metadata: Incomplete | None = ..., **kwargs: Any) -> None: ...

def reads(
    text: str,
    fmt: str = ...,
    as_version: int = ...,
    config: JupytextConfiguration | None = ...,
    **kwargs: Any,
) -> NotebookNode: ...
def read(
    fp: str | IO[Any], as_version: int = ..., fmt: str = ..., config: JupytextConfiguration | None = ..., **kwargs: Any
) -> NotebookNode: ...
def writes(
    notebook: NotebookNode, fmt: str, version: int = ..., config: JupytextConfiguration | None = ..., **kwargs: Any
) -> None: ...
def drop_text_representation_metadata(notebook: NotebookNode, metadata: Any = ...) -> Any: ...
def write(
    nb: NotebookNode,
    fp: str,
    version: int | None = ...,
    fmt: str = ...,
    config: JupytextConfiguration | None = ...,
    **kwargs: Any,
) -> None: ...
def create_prefix_dir(nb_file: str, fmt: str) -> None: ...
