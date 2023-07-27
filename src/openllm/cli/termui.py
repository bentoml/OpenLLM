from __future__ import annotations
import os
import typing as t

import click
import inflection

from ..utils import get_debug_mode
from ..utils import get_quiet_mode

def echo(text: t.Any, fg: str = "green", _with_style: bool = True, **attrs: t.Any) -> None:
    attrs["fg"], call = fg if not get_debug_mode() else None, click.echo if not _with_style else click.secho
    if not get_quiet_mode(): call(text, **attrs)


COLUMNS = int(os.getenv("COLUMNS", str(120)))

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": COLUMNS, "token_normalize_func": inflection.underscore}


__all__ = ["echo", "COLUMNS", "CONTEXT_SETTINGS"]
