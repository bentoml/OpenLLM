from typing import Any
from _typeshed import Incomplete

class JupytextFormatError(ValueError): ...

class NotebookFormatDescription:
    format_name: Incomplete
    extension: Incomplete
    header_prefix: Incomplete
    header_suffix: Incomplete
    cell_reader_class: Incomplete
    cell_exporter_class: Incomplete
    current_version_number: Incomplete
    min_readable_version_number: Incomplete
    def __init__(
        self,
        format_name: str,
        extension: str,
        header_prefix: str,
        cell_reader_class: Any,
        cell_exporter_class: Any,
        current_version_number: int,
        header_suffix: str = ...,
        min_readable_version_number: Incomplete | None = ...,
    ) -> None: ...

JUPYTEXT_FORMATS: Incomplete
NOTEBOOK_EXTENSIONS: Incomplete
EXTENSION_PREFIXES: Incomplete

def get_format_implementation(ext: str, format_name: str = ...) -> NotebookFormatDescription: ...
def guess_format(text: str, ext: str) -> tuple[str, dict[str, Any]]: ...
