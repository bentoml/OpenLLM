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
        format_name,
        extension,
        header_prefix,
        cell_reader_class,
        cell_exporter_class,
        current_version_number,
        header_suffix=...,
        min_readable_version_number=...,
    ) -> None: ...

JUPYTEXT_FORMATS: list[NotebookFormatDescription] = ...
NOTEBOOK_EXTENSIONS: list[str] = ...
EXTENSION_PREFIXES: list[str] = ...

def get_format_implementation(ext: str, format_name: str = ...) -> NotebookFormatDescription:
    """Return the implementation for the desired format"""
    ...

def read_metadata(text: str, ext: str) -> Any:
    """Return the header metadata"""
    ...

def read_format_from_metadata(text: str, ext: str) -> str | None:
    """Return the format of the file, when that information is available from the metadata"""
    ...

def guess_format(text: str, ext: str) -> tuple[str, dict[str, Any]]:
    """Guess the format and format options of the file, given its extension and content"""
    ...

def divine_format(text):  # -> Literal['ipynb', 'md']:
    """Guess the format of the notebook, based on its content #148"""
    ...

def check_file_version(notebook, source_path, outputs_path):  # -> None:
    """Raise if file version in source file would override outputs"""
    ...

def format_name_for_ext(metadata, ext, cm_default_formats=..., explicit_default=...):  # -> str | None:
    """Return the format name for that extension"""
    ...

def identical_format_path(fmt1, fmt2):  # -> bool:
    """Do the two (long representation) of formats target the same file?"""
    ...

def update_jupytext_formats_metadata(metadata, new_format):  # -> None:
    """Update the jupytext_format metadata in the Jupyter notebook"""
    ...

def rearrange_jupytext_metadata(metadata):  # -> None:
    """Convert the jupytext_formats metadata entry to jupytext/formats, etc. See #91"""
    ...

def long_form_one_format(
    jupytext_format, metadata=..., update=..., auto_ext_requires_language_info=...
):  # -> dict[Unknown, Unknown]:
    """Parse 'sfx.py:percent' into {'suffix':'sfx', 'extension':'py', 'format_name':'percent'}"""
    ...

def long_form_multiple_formats(
    jupytext_formats, metadata=..., auto_ext_requires_language_info=...
):  # -> list[Unknown] | list[dict[Unknown, Unknown]]:
    """Convert a concise encoding of jupytext.formats to a list of formats, encoded as dictionaries"""
    ...

def short_form_one_format(jupytext_format):
    """Represent one jupytext format as a string"""
    ...

def short_form_multiple_formats(jupytext_formats):  # -> LiteralString:
    """Convert jupytext formats, represented as a list of dictionaries, to a comma separated list"""
    ...

_VALID_FORMAT_INFO = ...
_BINARY_FORMAT_OPTIONS = ...
_VALID_FORMAT_OPTIONS = ...
_VALID_FORMAT_NAMES = ...

def validate_one_format(jupytext_format):  # -> dict[Unknown, Unknown]:
    """Validate extension and options for the given format"""
    ...

def auto_ext_from_metadata(metadata):  # -> str:
    """Script extension from notebook metadata"""
    ...

def check_auto_ext(fmt, metadata, option):
    """Replace the auto extension with the actual file extension, and raise a ValueError if it cannot be determined"""
    ...
