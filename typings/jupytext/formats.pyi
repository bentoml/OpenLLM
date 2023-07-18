"""
In this file the various text notebooks formats are defined. Please contribute
new formats here!
"""

from typing import Any

class JupytextFormatError(ValueError):
    """Error in the specification of the format for the text notebook"""

class NotebookFormatDescription:
    """Description of a notebook format"""

    def __init__(
        self,
        format_name: str,
        extension: str,
        header_prefix: str,
        cell_reader_class: Any,
        cell_exporter_class: Any,
        current_version_number: int,
        header_suffix: str = ...,
        min_readable_version_number: int = ...,
    ) -> None: ...

JUPYTEXT_FORMATS: list[NotebookFormatDescription] = ...
NOTEBOOK_EXTENSIONS: list[str] = ...
EXTENSION_PREFIXES: list[str] = ...

def get_format_implementation(ext: str, format_name: str = ...) -> NotebookFormatDescription:
    """Return the implementation for the desired format"""
    ...

def guess_format(text: str, ext: str) -> tuple[str, dict[str, Any]]:
    """Guess the format and format options of the file, given its extension and content"""
    ...
