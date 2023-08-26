from collections.abc import Generator
from typing import Any, Dict

from _typeshed import Incomplete

from .formats import NOTEBOOK_EXTENSIONS as NOTEBOOK_EXTENSIONS
class JupytextConfigurationError(ValueError): ...

JUPYTEXT_CONFIG_FILES: Incomplete
PYPROJECT_FILE: str
JUPYTEXT_CEILING_DIRECTORIES: Incomplete

class JupytextConfiguration:
    formats: Incomplete
    default_jupytext_formats: Incomplete
    preferred_jupytext_formats_save: Incomplete
    preferred_jupytext_formats_read: Incomplete
    notebook_metadata_filter: Incomplete
    default_notebook_metadata_filter: Incomplete
    hide_notebook_metadata: Incomplete
    root_level_metadata_as_raw_cell: Incomplete
    cell_metadata_filter: Incomplete
    default_cell_metadata_filter: Incomplete
    comment_magics: Incomplete
    split_at_heading: Incomplete
    sphinx_convert_rst2md: Incomplete
    doxygen_equation_markers: Incomplete
    outdated_text_notebook_margin: Incomplete
    cm_config_log_level: Incomplete
    cell_markers: Incomplete
    default_cell_markers: Incomplete
    notebook_extensions: Incomplete
    custom_cell_magics: Incomplete
    def set_default_format_options(self, format_options: Any, read: bool = ...) -> None: ...
    def default_formats(self, path: str) -> Any: ...
    def __eq__(self, other: object) -> bool: ...

def preferred_format(incomplete_format: Any, preferred_formats: Any) -> Any: ...
def global_jupytext_configuration_directories() -> Generator[Incomplete, Incomplete, None]: ...
def find_global_jupytext_configuration_file() -> Any: ...
def find_jupytext_configuration_file(path: str, search_parent_dirs: bool = ...) -> Any: ...
def parse_jupytext_configuration_file(jupytext_config_file: str, stream: Incomplete | None = ...) -> Any: ...
def load_jupytext_configuration_file(config_file: str, stream: Incomplete | None = ...) -> Any: ...
def load_jupytext_config(nb_file: str) -> JupytextConfiguration: ...
def validate_jupytext_configuration_file(config_file: str, config_dict: Dict[str, Any]) -> None: ...
def notebook_formats(
    nbk: Any, config: JupytextConfiguration, path: str, fallback_on_current_fmt: bool = ...
) -> Incomplete: ...
