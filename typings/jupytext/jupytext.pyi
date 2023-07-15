from typing import Any, IO
from nbformat.v4.rwbase import NotebookReader, NotebookWriter
from nbformat import NotebookNode
from .config import JupytextConfiguration

class NotSupportedNBFormatVersion(NotImplementedError):
    """An error issued when the current notebook format is not supported by this version of Jupytext"""

class TextNotebookConverter(NotebookReader, NotebookWriter):
    """A class that can read or write a Jupyter notebook as text"""

    def __init__(self, fmt, config) -> None: ...
    def update_fmt_with_notebook_options(self, metadata, read=...):  # -> None:
        """Update format options with the values in the notebook metadata, and record those
        options in the notebook metadata"""
        ...
    def reads(self, s, **_):
        """Read a notebook represented as text"""
        ...
    def filter_notebook(self, nb, metadata): ...
    def writes(self, nb, metadata=..., **kwargs):
        """Return the text representation of the notebook"""
        ...

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
