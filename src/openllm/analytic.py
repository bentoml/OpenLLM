from __future__ import annotations

import os, re, time, typing, abc
import attr, typer, typer.core

DO_NOT_TRACK = 'BENTOML_DO_NOT_TRACK'


class EventMeta(abc.ABC):
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
  def list_commands(self, ctx: typer.Context) -> list[str]:
    return list(self.commands)


class AnalyticsCommand(typer.core.TyperCommand):
  def invoke(self, ctx: typer.Context) -> typing.Any:
    if os.environ.get(DO_NOT_TRACK, 'false').lower() == 'true':
      return super().invoke(ctx)

    from bentoml._internal.utils.analytics import track

    command_name = ctx.info_name
    command_group = ctx.parent.info_name if ctx.parent and ctx.parent.parent else 'openllm'
    start_time = time.time_ns()
    try:
      result = super().invoke(ctx)
    except BaseException as e:
      track(
        OpenllmCliEvent(
          cmd_group=command_group,
          cmd_name=command_name,
          duration_in_ms=(time.time_ns() - start_time) / 1e6,
          error_type=type(e).__name__,
          return_code=2 if isinstance(e, KeyboardInterrupt) else 1,
        )
      )
      raise

    track(
      OpenllmCliEvent(
        cmd_group=command_group,
        cmd_name=command_name,
        duration_in_ms=(time.time_ns() - start_time) / 1e6,
      )
    )
    return result


class OpenLLMTyper(typer.Typer):
  def __init__(self, *args: typing.Any, **kwargs: typing.Any):
    no_args_is_help: bool = kwargs.pop('no_args_is_help', True)
    context_settings: dict[str, typing.Any] = kwargs.pop('context_settings', {})
    if 'help_option_names' not in context_settings:
      context_settings['help_option_names'] = ('-h', '--help')
    if 'max_content_width' not in context_settings:
      context_settings['max_content_width'] = int(os.environ.get('COLUMNS', str(120)))
    klass = kwargs.pop('cls', OrderedCommands)

    super().__init__(
      *args, cls=klass, no_args_is_help=no_args_is_help, context_settings=context_settings, **kwargs
    )

  if typing.TYPE_CHECKING:
    command = typer.Typer.command
  else:
    def command(self, *args: typing.Any, **kwargs: typing.Any):
      kwargs.setdefault('cls', AnalyticsCommand)
      return super().command(*args, **kwargs)
