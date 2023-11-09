from __future__ import annotations
import enum
import functools
import logging
import os
import typing as t

import click
import inflection
import orjson

from openllm_core._typing_compat import DictStrAny
from openllm_core.utils import get_debug_mode
from openllm_core.utils import get_quiet_mode


logger = logging.getLogger('openllm')


class Level(enum.IntEnum):
  NOTSET = logging.DEBUG
  DEBUG = logging.DEBUG
  INFO = logging.INFO
  WARNING = logging.WARNING
  ERROR = logging.ERROR
  CRITICAL = logging.CRITICAL

  @property
  def color(self) -> str | None:
    return {
      Level.NOTSET: None,
      Level.DEBUG: 'cyan',
      Level.INFO: 'green',
      Level.WARNING: 'yellow',
      Level.ERROR: 'red',
      Level.CRITICAL: 'red',
    }[self]


class JsonLog(t.TypedDict):
  log_level: Level
  content: str


def log(content: str, level: Level = Level.INFO, fg: str | None = None) -> None:
  def caller(text: str) -> None:
    if get_debug_mode():
      logger.log(level.value, text)
    else:
      echo(JsonLog(log_level=level, content=content), json=True, fg=fg)

  caller(orjson.dumps(JsonLog(log_level=level, content=content)).decode())


warning = functools.partial(log, level=Level.WARNING)
error = functools.partial(log, level=Level.ERROR)
critical = functools.partial(log, level=Level.CRITICAL)
debug = functools.partial(log, level=Level.DEBUG)
info = functools.partial(log, level=Level.INFO)
notset = functools.partial(log, level=Level.NOTSET)


def echo(text: t.Any, fg: str | None = None, _with_style: bool = True, json: bool = False, **attrs: t.Any) -> None:
  if json and not isinstance(text, dict):
    raise TypeError('text must be a dict')
  if json:
    if 'content' in text and 'log_level' in text:
      content = t.cast(DictStrAny, text)['content']
      fg = t.cast(Level, text['log_level']).color
    else:
      content = orjson.dumps(text).decode()
      fg = Level.INFO.color if not get_debug_mode() else Level.DEBUG.color
  else:
    content = t.cast(str, text)
  attrs['fg'] = fg if not get_debug_mode() else None

  if not get_quiet_mode():
    t.cast(t.Callable[..., None], click.echo if not _with_style else click.secho)(content, **attrs)


COLUMNS: int = int(os.environ.get('COLUMNS', str(120)))
CONTEXT_SETTINGS: DictStrAny = {
  'help_option_names': ['-h', '--help'],
  'max_content_width': COLUMNS,
  'token_normalize_func': inflection.underscore,
}
__all__ = ['echo', 'COLUMNS', 'CONTEXT_SETTINGS', 'log', 'warning', 'error', 'critical', 'debug', 'info', 'Level']
