from __future__ import annotations
import enum, functools, logging, os, typing as t
import click, inflection, orjson
from openllm_core._typing_compat import DictStrAny, TypedDict
from openllm_core.utils import get_debug_mode

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

  @classmethod
  def from_logging_level(cls, level: int) -> Level:
    return {
      logging.DEBUG: Level.DEBUG,
      logging.INFO: Level.INFO,
      logging.WARNING: Level.WARNING,
      logging.ERROR: Level.ERROR,
      logging.CRITICAL: Level.CRITICAL,
    }[level]


class JsonLog(TypedDict):
  log_level: Level
  content: str


def log(content: str, level: Level = Level.INFO, fg: str | None = None) -> None:
  if get_debug_mode():
    echo(content, fg=fg)
  else:
    echo(orjson.dumps(JsonLog(log_level=level, content=content)).decode(), fg=fg, json=True)


warning = functools.partial(log, level=Level.WARNING)
error = functools.partial(log, level=Level.ERROR)
critical = functools.partial(log, level=Level.CRITICAL)
debug = functools.partial(log, level=Level.DEBUG)
info = functools.partial(log, level=Level.INFO)
notset = functools.partial(log, level=Level.NOTSET)


def echo(text: t.Any, fg: str | None = None, *, _with_style: bool = True, json: bool = False, **attrs: t.Any) -> None:
  if json:
    text = orjson.loads(text)
    if 'content' in text and 'log_level' in text:
      content = text['content']
      fg = Level.from_logging_level(text['log_level']).color
    else:
      content = orjson.dumps(text).decode()
      fg = Level.INFO.color if not get_debug_mode() else Level.DEBUG.color
  else:
    content = t.cast(str, text)
  attrs['fg'] = fg

  (click.echo if not _with_style else click.secho)(content, **attrs)


COLUMNS: int = int(os.environ.get('COLUMNS', str(120)))
CONTEXT_SETTINGS: DictStrAny = {
  'help_option_names': ['-h', '--help'],
  'max_content_width': COLUMNS,
  'token_normalize_func': inflection.underscore,
}
__all__ = ['COLUMNS', 'CONTEXT_SETTINGS', 'Level', 'critical', 'debug', 'echo', 'error', 'info', 'log', 'warning']
