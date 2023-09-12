from __future__ import annotations
import os
import typing as t

import click
import inflection

import openllm

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import DictStrAny

def echo(text: t.Any, fg: str = 'green', _with_style: bool = True, **attrs: t.Any) -> None:
  attrs['fg'] = fg if not openllm.utils.get_debug_mode() else None
  if not openllm.utils.get_quiet_mode():
    t.cast(t.Callable[..., None], click.echo if not _with_style else click.secho)(text, **attrs)

COLUMNS: int = int(os.environ.get('COLUMNS', str(120)))
CONTEXT_SETTINGS: DictStrAny = {'help_option_names': ['-h', '--help'], 'max_content_width': COLUMNS, 'token_normalize_func': inflection.underscore}
__all__ = ['echo', 'COLUMNS', 'CONTEXT_SETTINGS']
