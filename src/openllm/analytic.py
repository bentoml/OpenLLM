from __future__ import annotations

import functools
import os
import re
import time
import typing
from abc import ABC

import attr
import click
import typer
import typer.core

DO_NOT_TRACK = 'BENTOML_DO_NOT_TRACK'


class EventMeta(ABC):
  @property
  def event_name(self):
    # camel case to snake case
    event_name = re.sub(r'(?<!^)(?=[A-Z])', '_', self.__class__.__name__).lower()
    # remove "_event" suffix
    suffix_to_remove = '_event'
    if event_name.endswith(suffix_to_remove):
      event_name = event_name[: -len(suffix_to_remove)]
    return event_name


@attr.define
class CliEvent(EventMeta):
  cmd_group: str
  cmd_name: str
  duration_in_ms: float = attr.field(default=0)
  error_type: typing.Optional[str] = attr.field(default=None)
  return_code: typing.Optional[int] = attr.field(default=None)


@attr.define
class OpenllmCliEvent(CliEvent):
  pass


class OrderedCommands(typer.core.TyperGroup):
  def list_commands(self, _: click.Context) -> typing.Iterable[str]:
    return list(self.commands)


class OpenLLMTyper(typer.Typer):
  def __init__(self, *args: typing.Any, **kwargs: typing.Any):
    no_args_is_help = kwargs.pop('no_args_is_help', True)
    context_settings = kwargs.pop('context_settings', {})
    if 'help_option_names' not in context_settings:
      context_settings['help_option_names'] = ('-h', '--help')
    if 'max_content_width' not in context_settings:
      context_settings['max_content_width'] = int(os.environ.get('COLUMNS', str(120)))
    klass = kwargs.pop('cls', OrderedCommands)

    super().__init__(*args, cls=klass, no_args_is_help=no_args_is_help, context_settings=context_settings, **kwargs)

  def command(self, *args: typing.Any, **kwargs: typing.Any):
    def decorator(f):
      @functools.wraps(f)
      @click.pass_context
      def wrapped(ctx: click.Context, *args, **kwargs):
        from bentoml._internal.utils.analytics import track

        do_not_track = os.environ.get(DO_NOT_TRACK, str(False)).lower() == 'true'

        # so we know that the root program is openllm
        command_name = ctx.info_name
        if ctx.parent.parent is not None:
          # openllm model list
          command_group = ctx.parent.info_name
        elif ctx.parent.info_name == ctx.find_root().info_name:
          # openllm run
          command_group = 'openllm'

        if do_not_track:
          return f(*args, **kwargs)
        start_time = time.time_ns()
        try:
          return_value = f(*args, **kwargs)
          duration_in_ns = time.time_ns() - start_time
          track(OpenllmCliEvent(cmd_group=command_group, cmd_name=command_name, duration_in_ms=duration_in_ns / 1e6))
          return return_value
        except BaseException as e:
          duration_in_ns = time.time_ns() - start_time
          track(
            OpenllmCliEvent(
              cmd_group=command_group,
              cmd_name=command_name,
              duration_in_ms=duration_in_ns / 1e6,
              error_type=type(e).__name__,
              return_code=2 if isinstance(e, KeyboardInterrupt) else 1,
            )
          )
          raise

      return typer.Typer.command(self, *args, **kwargs)(wrapped)

    return decorator
