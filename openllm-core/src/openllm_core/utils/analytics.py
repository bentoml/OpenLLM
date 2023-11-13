from __future__ import annotations
import contextlib
import functools
import importlib.metadata
import logging
import os
import re
import typing as t

import attr

import openllm_core
from openllm_core._typing_compat import ParamSpec

P = ParamSpec('P')
T = t.TypeVar('T')
logger = logging.getLogger(__name__)

# This variable is a proxy that will control BENTOML_DO_NOT_TRACK
OPENLLM_DO_NOT_TRACK = 'OPENLLM_DO_NOT_TRACK'


@functools.lru_cache(maxsize=1)
def do_not_track() -> bool:
  return openllm_core.utils.check_bool_env(OPENLLM_DO_NOT_TRACK)


@functools.lru_cache(maxsize=1)
def _usage_event_debugging() -> bool:
  return os.environ.get('__BENTOML_DEBUG_USAGE', str(False)).lower() == 'true'


def silent(func: t.Callable[P, T]) -> t.Callable[P, T]:
  @functools.wraps(func)
  def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
    try:
      return func(*args, **kwargs)
    except Exception as err:
      if _usage_event_debugging():
        if openllm_core.utils.get_debug_mode():
          logger.error('Tracking Error: %s', err, stack_info=True, stacklevel=3)
        else:
          logger.info('Tracking Error: %s', err)
      else:
        logger.debug('Tracking Error: %s', err)

  return wrapper


@silent
def track(event_properties: attr.AttrsInstance) -> None:
  from bentoml._internal.utils import analytics

  if do_not_track():
    return
  analytics.track(t.cast('analytics.schemas.EventMeta', event_properties))


@contextlib.contextmanager
def set_bentoml_tracking() -> t.Generator[None, None, None]:
  from bentoml._internal.utils import analytics

  original_value = os.environ.pop(analytics.BENTOML_DO_NOT_TRACK, str(False))
  try:
    os.environ[analytics.BENTOML_DO_NOT_TRACK] = str(do_not_track())
    yield
  finally:
    os.environ[analytics.BENTOML_DO_NOT_TRACK] = original_value


class EventMeta:
  @property
  def event_name(self) -> str:
    # camel case to snake case
    event_name = re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()
    # remove "_event" suffix
    suffix_to_remove = '_event'
    if event_name.endswith(suffix_to_remove):
      event_name = event_name[: -len(suffix_to_remove)]
    return event_name


@attr.define
class ModelSaveEvent(EventMeta):
  module: str
  model_size_in_kb: float


@attr.define
class OpenllmCliEvent(EventMeta):
  cmd_group: str
  cmd_name: str
  openllm_version: str = importlib.metadata.version('openllm')
  # NOTE: reserved for the do_not_track logics
  duration_in_ms: t.Any = attr.field(default=None)
  error_type: str = attr.field(default=None)
  return_code: int = attr.field(default=None)


@attr.define
class StartInitEvent(EventMeta):
  model_name: str
  llm_config: t.Dict[str, t.Any] = attr.field(default=None)

  @staticmethod
  def handler(llm_config: openllm_core.LLMConfig) -> StartInitEvent:
    return StartInitEvent(model_name=llm_config['model_name'], llm_config=llm_config.model_dump())


def track_start_init(llm_config: openllm_core.LLMConfig) -> None:
  if do_not_track():
    return
  track(StartInitEvent.handler(llm_config))
