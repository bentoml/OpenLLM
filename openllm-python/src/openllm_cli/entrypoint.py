from __future__ import annotations
import enum, functools, inspect, itertools, logging, os, platform, random, subprocess, threading, time, traceback, typing as t
import attr, click, fs, inflection, bentoml, openllm, orjson, fs.copy, fs.errors, click_option_group as cog
from bentoml_cli.utils import BentoMLCommandGroup, opt_callback
from simple_di import Provide, inject
from bentoml._internal.cloud.config import CloudClientConfig
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.model import ModelStore
from openllm import bundle
from openllm_core._typing_compat import (
  Concatenate,
  DictStrAny,
  LiteralBackend,
  LiteralDtype,
  LiteralQuantise,
  LiteralSerialisation,
  LiteralString,
  NotRequired,
  ParamSpec,
  Self,
  TypeGuard,
)
from openllm_core.config import CONFIG_MAPPING
from openllm_core.exceptions import OpenLLMException
from openllm_core.utils import (
  DEBUG_ENV_VAR,
  OPTIONAL_DEPENDENCIES,
  QUIET_ENV_VAR,
  LazyLoader,
  analytics,
  check_bool_env,
  compose,
  configure_logging,
  first_not_none,
  gen_random_uuid,
  get_debug_mode,
  get_quiet_mode,
  is_torch_available,
  pkg,
  resolve_user_filepath,
  set_debug_mode,
  set_quiet_mode,
)

from . import termui
from ._factory import FC, _AnyCallable, machine_option, model_name_argument, parse_config_options, start_decorator, optimization_decorator

if t.TYPE_CHECKING:
  import torch

  from bentoml._internal.bento import BentoStore
  from bentoml._internal.container import DefaultBuilder
  from openllm_client._schemas import StreamingResponse
  from openllm_core._configuration import LLMConfig
else:
  torch = LazyLoader('torch', globals(), 'torch')

P = ParamSpec('P')
logger = logging.getLogger('openllm')
OPENLLM_FIGLET = """\
 ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝
"""

ServeCommand = t.Literal['serve', 'serve-grpc']


@attr.define
class GlobalOptions:
  cloud_context: str | None = attr.field(default=None)

  def with_options(self, **attrs: t.Any) -> Self:
    return attr.evolve(self, **attrs)


GrpType = t.TypeVar('GrpType', bound=click.Group)

_object_setattr = object.__setattr__

_EXT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'extension'))


def backend_warning(backend: LiteralBackend, build: bool = False) -> None:
  if backend == 'pt':
    if openllm.utils.is_vllm_available():
      logger.warning(
        'vLLM is available, but using PyTorch backend instead. Note that vLLM is a lot more performant and should always be used in production (by explicitly set --backend vllm).'
      )
    else:
      logger.warning(
        'vLLM is not available. Note that PyTorch backend is not as performant as vLLM and you should always consider using vLLM for production.'
      )
    if build:
      logger.info("Tip: You can set '--backend vllm' to package your Bento with vLLM backend regardless if vLLM is available locally.")


class Extensions(click.MultiCommand):
  def list_commands(self, ctx: click.Context) -> list[str]:
    return sorted([filename[:-3] for filename in os.listdir(_EXT_FOLDER) if filename.endswith('.py') and not filename.startswith('__')])

  def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
    try:
      mod = __import__(f'openllm_cli.extension.{cmd_name}', None, None, ['cli'])
    except ImportError:
      return None
    return mod.cli


class OpenLLMCommandGroup(BentoMLCommandGroup):
  NUMBER_OF_COMMON_PARAMS = 5  # parameters in common_params + 1 faked group option header

  @staticmethod
  def common_params(f: t.Callable[P, t.Any]) -> t.Callable[[FC], FC]:
    # The following logics is similar to one of BentoMLCommandGroup
    @cog.optgroup.group(name='Global options', help='Shared globals options for all OpenLLM CLI.')  # type: ignore[misc]
    @cog.optgroup.option('-q', '--quiet', envvar=QUIET_ENV_VAR, is_flag=True, default=False, help='Suppress all output.', show_envvar=True)
    @cog.optgroup.option(
      '--debug', '--verbose', 'debug', envvar=DEBUG_ENV_VAR, is_flag=True, default=False, help='Print out debug logs.', show_envvar=True
    )
    @cog.optgroup.option(
      '--do-not-track', is_flag=True, default=False, envvar=analytics.OPENLLM_DO_NOT_TRACK, help='Do not send usage info', show_envvar=True
    )
    @cog.optgroup.option(
      '--context', 'cloud_context', envvar='BENTOCLOUD_CONTEXT', type=click.STRING, default=None, help='BentoCloud context name.', show_envvar=True
    )
    @click.pass_context
    @functools.wraps(f)
    def wrapper(ctx: click.Context, quiet: bool, debug: bool, cloud_context: str | None, *args: P.args, **attrs: P.kwargs) -> t.Any:
      ctx.obj = GlobalOptions(cloud_context=cloud_context)
      if quiet:
        set_quiet_mode(True)
        if debug:
          termui.warning("'--quiet' passed; ignoring '--verbose/--debug'")
      elif debug:
        set_debug_mode(True)
      configure_logging()
      return f(*args, **attrs)

    return wrapper

  @staticmethod
  def usage_tracking(func: t.Callable[P, t.Any], group: click.Group, **attrs: t.Any) -> t.Callable[Concatenate[bool, P], t.Any]:
    command_name = attrs.get('name', func.__name__)

    @functools.wraps(func)
    def wrapper(do_not_track: bool, *args: P.args, **attrs: P.kwargs) -> t.Any:
      if do_not_track:
        with analytics.set_bentoml_tracking():
          return func(*args, **attrs)
      start_time = time.time_ns()
      with analytics.set_bentoml_tracking():
        if group.name is None:
          raise ValueError('group.name should not be None')
        event = analytics.OpenllmCliEvent(cmd_group=group.name, cmd_name=command_name)
        try:
          return_value = func(*args, **attrs)
          duration_in_ms = (time.time_ns() - start_time) / 1e6
          event.duration_in_ms = duration_in_ms
          analytics.track(event)
          return return_value
        except Exception as e:
          duration_in_ms = (time.time_ns() - start_time) / 1e6
          event.duration_in_ms = duration_in_ms
          event.error_type = type(e).__name__
          event.return_code = 2 if isinstance(e, KeyboardInterrupt) else 1
          analytics.track(event)
          raise

    return t.cast(t.Callable[Concatenate[bool, P], t.Any], wrapper)

  def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
    if cmd_name in t.cast('Extensions', extension_command).list_commands(ctx):
      return t.cast('Extensions', extension_command).get_command(ctx, cmd_name)
    cmd_name = self.resolve_alias(cmd_name)
    return super().get_command(ctx, cmd_name)

  def list_commands(self, ctx: click.Context) -> list[str]:
    return super().list_commands(ctx) + extension_command.list_commands(ctx)

  def command(self, *args: t.Any, **kwargs: t.Any) -> t.Callable[[t.Callable[..., t.Any]], click.Command]:
    """Override the default 'cli.command' with supports for aliases for given command, and it wraps the implementation with common parameters."""
    if 'context_settings' not in kwargs:
      kwargs['context_settings'] = {}
    if 'max_content_width' not in kwargs['context_settings']:
      kwargs['context_settings']['max_content_width'] = 120
    aliases = kwargs.pop('aliases', None)

    def decorator(f: _AnyCallable) -> click.Command:
      name = f.__name__.lower()
      if name.endswith('_command'):
        name = name[:-8]
      name = name.replace('_', '-')
      kwargs.setdefault('help', inspect.getdoc(f))
      kwargs.setdefault('name', name)
      wrapped = self.usage_tracking(self.common_params(f), self, **kwargs)

      # move common parameters to end of the parameters list
      _memo = getattr(wrapped, '__click_params__', None)
      if _memo is None:
        raise ValueError('Click command not register correctly.')
      _object_setattr(wrapped, '__click_params__', _memo[-self.NUMBER_OF_COMMON_PARAMS :] + _memo[: -self.NUMBER_OF_COMMON_PARAMS])
      # NOTE: we need to call super of super to avoid conflict with BentoMLCommandGroup command setup
      cmd = super(BentoMLCommandGroup, self).command(*args, **kwargs)(wrapped)
      # NOTE: add aliases to a given commands if it is specified.
      if aliases is not None:
        if not cmd.name:
          raise ValueError('name is required when aliases are available.')
        self._commands[cmd.name] = aliases
        self._aliases.update({alias: cmd.name for alias in aliases})
      return cmd

    return decorator

  def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
    """Additional format methods that include extensions as well as the default cli command."""
    from gettext import gettext as _

    commands: list[tuple[str, click.Command]] = []
    extensions: list[tuple[str, click.Command]] = []
    _cached_extensions: list[str] = t.cast('Extensions', extension_command).list_commands(ctx)
    for subcommand in self.list_commands(ctx):
      cmd = self.get_command(ctx, subcommand)
      if cmd is None or cmd.hidden:
        continue
      if subcommand in _cached_extensions:
        extensions.append((subcommand, cmd))
      else:
        commands.append((subcommand, cmd))
    # allow for 3 times the default spacing
    if len(commands):
      limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)
      rows: list[tuple[str, str]] = []
      for subcommand, cmd in commands:
        help = cmd.get_short_help_str(limit)
        rows.append((subcommand, help))
      if rows:
        with formatter.section(_('Commands')):
          formatter.write_dl(rows)
    if len(extensions):
      limit = formatter.width - 6 - max(len(cmd[0]) for cmd in extensions)
      rows = []
      for subcommand, cmd in extensions:
        help = cmd.get_short_help_str(limit)
        rows.append((inflection.dasherize(subcommand), help))
      if rows:
        with formatter.section(_('Extensions')):
          formatter.write_dl(rows)


_PACKAGE_NAME = 'openllm'


@click.group(cls=OpenLLMCommandGroup, context_settings=termui.CONTEXT_SETTINGS, name='openllm')
@click.version_option(
  None,
  '--version',
  '-v',
  package_name=_PACKAGE_NAME,
  message=f'{_PACKAGE_NAME}, %(version)s (compiled: {openllm.COMPILED})\nPython ({platform.python_implementation()}) {platform.python_version()}',
)
def cli() -> None:
  """\b
   ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
  ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
  ╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝.

  \b
  An open platform for operating large language models in production.
  Fine-tune, serve, deploy, and monitor any LLMs with ease.
  """


@cli.command(context_settings=termui.CONTEXT_SETTINGS, name='start', aliases=['start-http'], short_help='Start a LLMServer for any supported LLM.')
@click.argument('model_id', type=click.STRING, metavar='[REMOTE_REPO/MODEL_ID | /path/to/local/model]', required=True)
@click.option(
  '--model-id',
  'deprecated_model_id',
  type=click.STRING,
  default=None,
  hidden=True,
  metavar='[REMOTE_REPO/MODEL_ID | /path/to/local/model]',
  help='Deprecated. Use positional argument instead.',
)
@click.option(
  '--max-model-len',
  '--max_model_len',
  'max_model_len',
  type=int,
  default=None,
  help='Maximum sequence length for the model. If not specified, we will use the default value from the model config.',
)
@click.option(
  '--gpu-memory-utilization',
  '--gpu_memory_utilization',
  'gpu_memory_utilization',
  default=0.9,
  help='The percentage of GPU memory to be used for the model executor',
)
@start_decorator
def start_command(
  model_id: str,
  server_timeout: int,
  model_version: str | None,
  workers_per_resource: t.Literal['conserved', 'round_robin'] | LiteralString,
  device: t.Tuple[str, ...],
  quantize: LiteralQuantise | None,
  backend: LiteralBackend | None,
  serialisation: LiteralSerialisation | None,
  cors: bool,
  adapter_id: str | None,
  return_process: bool,
  dtype: LiteralDtype,
  deprecated_model_id: str | None,
  max_model_len: int | None,
  gpu_memory_utilization: float,
  **attrs: t.Any,
) -> LLMConfig | subprocess.Popen[bytes]:
  """Start any LLM as a REST server.

  \b
  ```bash
  $ openllm <start|start-http> <model_id> --<options> ...
  ```
  """
  if backend == 'pt':
    logger.warning('PyTorch backend is deprecated and will be removed in future releases. Make sure to use vLLM instead.')
  if model_id in openllm.CONFIG_MAPPING:
    _model_name = model_id
    if deprecated_model_id is not None:
      model_id = deprecated_model_id
    else:
      model_id = openllm.AutoConfig.for_model(_model_name)['default_id']
    logger.warning(
      "Passing 'openllm start %s%s' is deprecated and will be remove in a future version. Use 'openllm start %s' instead.",
      _model_name,
      '' if deprecated_model_id is None else f' --model-id {deprecated_model_id}',
      model_id,
    )

  adapter_map: dict[str, str] | None = attrs.pop('adapter_map', None)

  from openllm.serialisation.transformers.weights import has_safetensors_weights

  serialisation = first_not_none(serialisation, default='safetensors' if has_safetensors_weights(model_id, model_version) else 'legacy')

  if serialisation == 'safetensors' and quantize is not None:
    logger.warning("'--quantize=%s' might not work with 'safetensors' serialisation format.", quantize)
    logger.warning("Make sure to check out '%s' repository to see if the weights is in '%s' format if unsure.", model_id, serialisation)
    logger.info("Tip: You can always fallback to '--serialisation legacy' when running quantisation.")

  import torch

  if backend == 'pt' and not torch.cuda.is_available():
    if dtype == 'auto':
      dtype = 'float'
    elif dtype not in {'float', 'float32'}:
      logger.warning('"bfloat16" and "half" are not supported on CPU. OpenLLM will default fallback to "float32".')
    dtype = 'float'  # we need to cast back to full precision if cuda is not available
  llm = openllm.LLM(
    model_id=model_id,
    model_version=model_version,
    backend=backend,
    adapter_map=adapter_map,
    quantize=quantize,
    serialisation=serialisation,
    dtype=dtype,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    trust_remote_code=check_bool_env('TRUST_REMOTE_CODE', False),
  )
  backend_warning(llm.__llm_backend__)

  config, server_attrs = llm.config.model_validate_click(**attrs)
  server_timeout = first_not_none(server_timeout, default=config['timeout'])
  server_attrs.update({'working_dir': pkg.source_locations('openllm'), 'timeout': server_timeout})
  development = server_attrs.pop('development')  # XXX: currently, theres no development args in bentoml.Server. To be fixed upstream.
  server_attrs.setdefault('production', not development)

  start_env = process_environ(
    config,
    server_timeout,
    process_workers_per_resource(first_not_none(workers_per_resource, default=config['workers_per_resource']), device),
    device,
    cors,
    model_id,
    adapter_map,
    serialisation,
    llm,
  )

  server = bentoml.HTTPServer('_service:svc', **server_attrs)
  openllm.utils.analytics.track_start_init(config)

  try:
    build_bento_instruction(llm, model_id, serialisation, adapter_map)
    it = run_server(server.args, start_env, return_process=return_process)
    if return_process:
      return it
  except KeyboardInterrupt:
    pass

  # NOTE: Return the configuration for telemetry purposes.
  return config


def process_environ(config, server_timeout, wpr, device, cors, model_id, adapter_map, serialisation, llm, use_current_env=True):
  environ = parse_config_options(config, server_timeout, wpr, device, cors, os.environ.copy() if use_current_env else {})
  environ.update({
    'OPENLLM_MODEL_ID': model_id,
    'BENTOML_DEBUG': str(openllm.utils.get_debug_mode()),
    'BENTOML_HOME': os.environ.get('BENTOML_HOME', BentoMLContainer.bentoml_home.get()),
    'OPENLLM_ADAPTER_MAP': orjson.dumps(adapter_map).decode(),
    'OPENLLM_SERIALIZATION': serialisation,
    'OPENLLM_CONFIG': config.model_dump_json(flatten=True).decode(),
    'BACKEND': llm.__llm_backend__,
    'DTYPE': str(llm._torch_dtype).split('.')[-1],
    'TRUST_REMOTE_CODE': str(llm.trust_remote_code),
    'MAX_MODEL_LEN': orjson.dumps(llm._max_model_len).decode(),
    'GPU_MEMORY_UTILIZATION': orjson.dumps(llm._gpu_memory_utilization).decode(),
  })
  if llm.quantise:
    environ['QUANTIZE'] = str(llm.quantise)
  return environ


def process_workers_per_resource(wpr: str | float | int, device: tuple[str, ...]) -> TypeGuard[float]:
  if isinstance(wpr, str):
    if wpr == 'round_robin':
      wpr = 1.0
    elif wpr == 'conserved':
      if device and openllm.utils.device_count() == 0:
        termui.echo('--device will have no effect as there is no GPUs available', fg='yellow')
        wpr = 1.0
      else:
        available_gpu = len(device) if device else openllm.utils.device_count()
        wpr = 1.0 if available_gpu == 0 else float(1 / available_gpu)
    else:
      wpr = float(wpr)
  elif isinstance(wpr, int):
    wpr = float(wpr)
  return wpr


def build_bento_instruction(llm, model_id, serialisation, adapter_map):
  cmd_name = f'openllm build {model_id} --backend {llm.__llm_backend__}'
  if llm.quantise:
    cmd_name += f' --quantize {llm.quantise}'
  if llm.__llm_backend__ in {'pt', 'vllm'}:
    cmd_name += f' --serialization {serialisation}'
  if adapter_map is not None:
    cmd_name += ' ' + ' '.join([
      f'--adapter-id {s}' for s in [f'{p}:{name}' if name not in (None, 'default') else p for p, name in adapter_map.items()]
    ])
  if not openllm.utils.get_quiet_mode():
    termui.info(f"🚀Tip: run '{cmd_name}' to create a BentoLLM for '{model_id}'")


def pretty_print(line: str):
  if 'WARNING' in line:
    caller = termui.warning
  elif 'INFO' in line:
    caller = termui.info
  elif 'DEBUG' in line:
    caller = termui.debug
  elif 'ERROR' in line:
    caller = termui.error
  else:
    caller = functools.partial(termui.echo, fg=None)
  caller(line.strip())


def handle(stream, stop_event):
  try:
    for line in iter(stream.readline, ''):
      if stop_event.is_set():
        break
      pretty_print(line)
  finally:
    stream.close()


def run_server(args, env, return_process=False) -> subprocess.Popen[bytes] | int:
  process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, text=True)
  if return_process:
    return process
  stop_event = threading.Event()
  stdout, stderr = threading.Thread(target=handle, args=(process.stdout, stop_event)), threading.Thread(target=handle, args=(process.stderr, stop_event))
  stdout.start(); stderr.start()  # noqa: E702

  try:
    process.wait()
  except KeyboardInterrupt:
    stop_event.set()
    process.terminate()
    try:
      process.wait(0.1)
    except subprocess.TimeoutExpired:
      # not sure if the process exits cleanly
      process.kill()
    raise
  finally:
    stop_event.set()
    stdout.join(); stderr.join()  # noqa: E702
    if process.poll() is not None: process.kill()
    stdout.join(); stderr.join()  # noqa: E702

  return process.returncode


class ItemState(enum.Enum):
  NOT_FOUND = 'NOT_FOUND'
  ADDED = 'ADDED'
  EXISTS = 'EXISTS'
  OVERWRITE = 'OVERWRITE'


class ImportModelOutput(t.TypedDict):
  state: ItemState
  backend: LiteralBackend
  tag: str


@cli.command(name='import', aliases=['download'])
@click.argument('model_id', type=click.STRING, metavar='[REMOTE_REPO/MODEL_ID | /path/to/local/model]', required=True)
@click.option(
  '--model-id',
  'deprecated_model_id',
  type=click.STRING,
  default=None,
  hidden=True,
  metavar='[REMOTE_REPO/MODEL_ID | /path/to/local/model]',
  help='Deprecated. Use positional argument instead.',
)
@optimization_decorator
def import_command(
  model_id: str,
  deprecated_model_id: str | None,
  dtype: LiteralDtype,
  model_version: str | None,
  backend: LiteralBackend | None,
  quantize: LiteralQuantise | None,
  serialisation: LiteralSerialisation | None,
) -> ImportModelOutput:
  """Setup LLM interactively.

  \b
  This `model_id` can be either pretrained model id that you can get from HuggingFace Hub, or
  a custom model path from your custom pretrained model. Note that the custom model path should
  contain all files required to construct `transformers.PretrainedConfig`, `transformers.PreTrainedModel`
  and `transformers.PreTrainedTokenizer` objects.

  \b
  Note that if `--serialisation` is not defined, then we will try to infer serialisation from HuggingFace Hub.
  If the model id contains safetensors weights, then we will use `safetensors` serialisation. Otherwise, we will
  fallback to `legacy` '.bin' (otherwise known as pickle) serialisation.

  \b
  ``--model-version`` is an optional option to save the model. Note that
  this is recommended when the model_id is a custom path. Usually, if you are only using pretrained
  model from HuggingFace Hub, you don't need to specify this. If this is not specified, we will calculate
  the hash from the last modified time from this custom path

  \b
  ```bash
  $ openllm import mistralai/Mistral-7B-v0.1
  ```

  \b
  > If ``quantize`` is passed, the model weights will be saved as quantized weights. You should
  > only use this option if you want the weight to be quantized by default. Note that OpenLLM also
  > support on-demand quantisation during initial startup.
  """
  from openllm.serialisation.transformers.weights import has_safetensors_weights

  if model_id in openllm.CONFIG_MAPPING:
    _model_name = model_id
    if deprecated_model_id is not None:
      model_id = deprecated_model_id
    else:
      model_id = openllm.AutoConfig.for_model(_model_name)['default_id']
    termui.echo(
      f"Passing 'openllm import {_model_name}{'' if deprecated_model_id is None else ' --model-id ' + deprecated_model_id}' is deprecated and will be remove in a future version. Use 'openllm import {model_id}' instead.",
      fg='yellow',
    )

  llm = openllm.LLM(
    model_id=model_id,
    model_version=model_version,
    quantize=quantize,
    backend=backend,
    dtype=dtype,
    serialisation=t.cast(
      LiteralSerialisation, first_not_none(serialisation, default='safetensors' if has_safetensors_weights(model_id, model_version) else 'legacy')
    ),
  )
  backend_warning(llm.__llm_backend__)

  state = ItemState.NOT_FOUND
  try:
    model = bentoml.models.get(llm.tag)
    state = ItemState.EXISTS
  except bentoml.exceptions.NotFound:
    model = openllm.serialisation.import_model(llm, trust_remote_code=llm.trust_remote_code)
    if llm.__llm_backend__ == 'pt' and is_torch_available() and torch.cuda.is_available():
      torch.cuda.empty_cache()
    state = ItemState.ADDED
  response = ImportModelOutput(state=state, backend=llm.__llm_backend__, tag=str(model.tag))
  termui.echo(orjson.dumps(response).decode(), fg='white')
  return response


@attr.define(auto_attribs=True)
class _Content:
  instr: str
  cmd: str

  def __str__(self) -> str:
    return self.instr.format(cmd=self.cmd)


@attr.define(auto_attribs=True)
class DeploymentInstruction:
  type: t.Literal['container', 'bentocloud']
  content: _Content

  @classmethod
  def from_content(cls, type: t.Literal['container', 'bentocloud'], instr: str, cmd: str) -> DeploymentInstruction:
    return cls(type=type, content=_Content(instr=instr, cmd=cmd))

  def __getitem__(self, key: str) -> str:
    return getattr(self, key)


class BuildBentoOutput(t.TypedDict):
  state: ItemState
  tag: str
  backend: LiteralBackend
  instructions: t.List[DeploymentInstruction]


@cli.command(context_settings={'token_normalize_func': inflection.underscore})
@click.argument('model_id', type=click.STRING, metavar='[REMOTE_REPO/MODEL_ID | /path/to/local/model]', required=True)
@click.option(
  '--model-id',
  'deprecated_model_id',
  type=click.STRING,
  default=None,
  hidden=True,
  metavar='[REMOTE_REPO/MODEL_ID | /path/to/local/model]',
  help='Deprecated. Use positional argument instead.',
)
@click.option('--bento-version', type=str, default=None, help='Optional bento version for this BentoLLM. Default is the the model revision.')
@click.option('--overwrite', is_flag=True, help='Overwrite existing Bento for given LLM if it already exists.')
@click.option(
  '--enable-features',
  multiple=True,
  nargs=1,
  metavar='FEATURE[,FEATURE]',
  help='Enable additional features for building this LLM Bento. Available: {}'.format(', '.join(OPTIONAL_DEPENDENCIES)),
)
@optimization_decorator
@click.option(
  '--adapter-id',
  default=None,
  multiple=True,
  metavar='[PATH | [remote/][adapter_name:]adapter_id][, ...]',
  help="Optional adapters id to be included within the Bento. Note that if you are using relative path, '--build-ctx' must be passed.",
)
@click.option('--build-ctx', help='Build context. This is required if --adapter-id uses relative path', default=None)
@click.option('--dockerfile-template', default=None, type=click.File(), help='Optional custom dockerfile template to be used with this BentoLLM.')
@cog.optgroup.group(cls=cog.MutuallyExclusiveOptionGroup, name='Utilities options')  # type: ignore[misc]
@cog.optgroup.option(
  '--containerize',
  default=False,
  is_flag=True,
  type=click.BOOL,
  help="Whether to containerize the Bento after building. '--containerize' is the shortcut of 'openllm build && bentoml containerize'.",
)
@cog.optgroup.option(
  '--push',
  default=False,
  is_flag=True,
  type=click.BOOL,
  help="Whether to push the result bento to BentoCloud. Make sure to login with 'bentoml cloud login' first.",
)
@click.option('--force-push', default=False, is_flag=True, type=click.BOOL, help='Whether to force push.')
@click.option(
  '--max-model-len',
  '--max_model_len',
  'max_model_len',
  default=None,
  help='Maximum sequence length for the model. If not specified, we will use the default value from the model config.',
)
@click.option(
  '--gpu-memory-utilization',
  '--gpu_memory_utilization',
  'gpu_memory_utilization',
  default=0.9,
  help='The percentage of GPU memory to be used for the model executor',
)
@machine_option
@click.pass_context
def build_command(
  ctx: click.Context,
  /,
  model_id: str,
  deprecated_model_id: str | None,
  bento_version: str | None,
  overwrite: bool,
  dtype: LiteralDtype,
  model_version: str | None,
  backend: LiteralBackend | None,
  quantize: LiteralQuantise | None,
  serialisation: LiteralSerialisation | None,
  machine: bool,
  enable_features: tuple[str, ...] | None,
  adapter_id: tuple[str, ...],
  build_ctx: str | None,
  dockerfile_template: t.TextIO | None,
  max_model_len: int | None,
  gpu_memory_utilization: float,
  containerize: bool,
  push: bool,
  force_push: bool,
  **_: t.Any,
) -> BuildBentoOutput:
  """Package a given models into a BentoLLM.

  \b
  ```bash
  $ openllm build google/flan-t5-large
  ```

  \b
  > [!NOTE]
  > To run a container built from this Bento with GPU support, make sure
  > to have https://github.com/NVIDIA/nvidia-container-toolkit install locally.

  \b
  > [!IMPORTANT]
  > To build the bento with compiled OpenLLM, make sure to prepend HATCH_BUILD_HOOKS_ENABLE=1. Make sure that the deployment
  > target also use the same Python version and architecture as build machine.
  """
  from openllm.serialisation.transformers.weights import has_safetensors_weights

  if model_id in openllm.CONFIG_MAPPING:
    _model_name = model_id
    if deprecated_model_id is not None:
      model_id = deprecated_model_id
    else:
      model_id = openllm.AutoConfig.for_model(_model_name)['default_id']
    termui.echo(
      f"Passing 'openllm build {_model_name}{'' if deprecated_model_id is None else ' --model-id ' + deprecated_model_id}' is deprecated and will be remove in a future version. Use 'openllm build {model_id}' instead.",
      fg='yellow',
    )

  if enable_features:
    enable_features = tuple(itertools.chain.from_iterable((s.split(',') for s in enable_features)))

  state = ItemState.NOT_FOUND

  if backend == 'pt':
    logger.warning("PyTorch backend is deprecated and will be removed from the next releases. Will set default backend to 'vllm' instead.")

  llm = openllm.LLM(
    model_id=model_id,
    model_version=model_version,
    backend=backend,
    quantize=quantize,
    dtype=dtype,
    max_model_len=max_model_len,
    gpu_memory_utilization=gpu_memory_utilization,
    serialisation=first_not_none(serialisation, default='safetensors' if has_safetensors_weights(model_id, model_version) else 'legacy'),
    _eager=False,
  )
  if llm.__llm_backend__ not in llm.config['backend']:
    raise click.ClickException(f"'{backend}' is not supported with {model_id}")
  backend_warning(llm.__llm_backend__, build=True)
  try:
    model = bentoml.models.get(llm.tag)
  except bentoml.exceptions.NotFound:
    model = openllm.serialisation.import_model(llm, trust_remote_code=llm.trust_remote_code)
  llm._tag = model.tag

  os.environ.update(**process_environ(llm.config, llm.config['timeout'], 1.0, None, True, llm.model_id, None, llm._serialisation, llm))

  try:
    assert llm.bentomodel  # HACK: call it here to patch correct tag with revision and everything

    labels = dict(llm.identifying_params)
    labels.update({'_type': llm.llm_type, '_framework': llm.__llm_backend__})

    with fs.open_fs(f'temp://llm_{gen_random_uuid()}') as llm_fs:
      dockerfile_template_path = None
      if dockerfile_template:
        with dockerfile_template:
          llm_fs.writetext('Dockerfile.template', dockerfile_template.read())
        dockerfile_template_path = llm_fs.getsyspath('/Dockerfile.template')

      adapter_map = None
      if adapter_id and not build_ctx:
        ctx.fail("'build_ctx' is required when '--adapter-id' is passsed.")
      if adapter_id:
        adapter_map = {}
        for v in adapter_id:
          _adapter_id, *adapter_name = v.rsplit(':', maxsplit=1)
          name = adapter_name[0] if len(adapter_name) > 0 else 'default'
          try:
            resolve_user_filepath(_adapter_id, build_ctx)
            src_folder_name = os.path.basename(_adapter_id)
            src_fs = fs.open_fs(build_ctx)
            llm_fs.makedir(src_folder_name, recreate=True)
            fs.copy.copy_dir(src_fs, _adapter_id, llm_fs, src_folder_name)
            adapter_map[src_folder_name] = name
          # this is the remote adapter, then just added back
          # note that there is a drawback here. If the path of the local adapter
          # path have the same name as the remote, then we currently don't support
          # that edge case.
          except FileNotFoundError:
            adapter_map[_adapter_id] = name
        os.environ['OPENLLM_ADAPTER_MAP'] = orjson.dumps(adapter_map).decode()

      _bento_version = first_not_none(bento_version, default=llm.bentomodel.tag.version)
      bento_tag = bentoml.Tag.from_taglike(f'{llm.llm_type}-service:{_bento_version}'.lower().strip())
      try:
        bento = bentoml.get(bento_tag)
        if overwrite:
          bentoml.delete(bento_tag)
          state = ItemState.OVERWRITE
          raise bentoml.exceptions.NotFound(f'Rebuilding existing Bento {bento_tag}') from None
        state = ItemState.EXISTS
      except bentoml.exceptions.NotFound:
        bento = bundle.create_bento(
          bento_tag,
          llm_fs,
          llm,
          adapter_map=adapter_map,
          quantize=quantize,
          extra_dependencies=enable_features,
          dockerfile_template=dockerfile_template_path,
        )
        if state != ItemState.OVERWRITE:
          state = ItemState.ADDED
  except Exception as err:
    traceback.print_exc()
    raise click.ClickException('Exception caught while building BentoLLM:\n' + str(err)) from err

  cloud_config = CloudClientConfig.get_config()

  def get_current_bentocloud_context() -> str | None:
    try:
      context = cloud_config.get_context(ctx.obj.cloud_context) if ctx.obj.cloud_context else cloud_config.get_current_context()
      return context.name
    except Exception:
      return None

  push_cmd = f'bentoml push {bento_tag}'
  cloud_context = get_current_bentocloud_context()
  if cloud_context is None:
    available_context = [c.name for c in cloud_config.contexts]
    if not available_context:
      logger.warning('No default BentoCloud context found. Please login with `bentoml cloud login` first.')
    else:
      logger.warning(
        'No context is passed, but the following context is available: %s. Make sure to specify the argument "--context" for specific context you want to push to.',
        available_context,
      )
  else:
    push_cmd += f' --context {cloud_context}'

  response = BuildBentoOutput(
    state=state,
    tag=str(bento_tag),
    backend=llm.__llm_backend__,
    instructions=[
      DeploymentInstruction.from_content(type='bentocloud', instr="☁️  Push to BentoCloud with 'bentoml push':\n    $ {cmd}", cmd=push_cmd),
      DeploymentInstruction.from_content(
        type='container',
        instr="🐳 Container BentoLLM with 'bentoml containerize':\n    $ {cmd}",
        cmd=f'bentoml containerize {bento_tag} --opt progress=plain',
      ),
    ],
  )

  plain_instruction = {i.type: i['content'].cmd for i in response['instructions']}
  if machine or get_debug_mode():
    response['instructions'] = plain_instruction
  if machine:
    termui.echo(f'__object__:{orjson.dumps(response).decode()}\n\n', fg='white')
  elif not get_quiet_mode() and (not push or not containerize):
    if state != ItemState.EXISTS:
      termui.info(f"Successfully built Bento '{bento.tag}'.\n")
    elif not overwrite:
      termui.warning(f"Bento for '{model_id}' already exists [{bento}]. To overwrite it pass '--overwrite'.\n")
    if not get_debug_mode():
      termui.echo(OPENLLM_FIGLET)
      termui.echo('📖 Next steps:\n', nl=False)
      for instruction in response['instructions']:
        termui.echo(f"  * {instruction['content']}\n", nl=False)

  if push:
    BentoMLContainer.bentocloud_client.get().push_bento(bento, context=t.cast(GlobalOptions, ctx.obj).cloud_context, force=force_push)
  elif containerize:
    container_backend = t.cast('DefaultBuilder', os.environ.get('BENTOML_CONTAINERIZE_BACKEND', 'docker'))
    try:
      bentoml.container.health(container_backend)
    except subprocess.CalledProcessError:
      raise OpenLLMException(f'Failed to use backend {backend}') from None
    try:
      bentoml.container.build(bento.tag, backend=container_backend, features=('grpc', 'io'))
    except Exception as err:
      raise OpenLLMException(f"Exception caught while containerizing '{bento.tag!s}':\n{err}") from err

  if get_debug_mode():
    termui.echo('\n' + orjson.dumps(response).decode(), fg=None)
  return response


class ModelItem(t.TypedDict):
  architecture: str
  example_id: str
  supported_backends: t.Tuple[LiteralBackend, ...]
  installation: str
  items: NotRequired[t.List[str]]


@cli.command()
@click.option('--show-available', is_flag=True, default=True, hidden=True)
def models_command(**_: t.Any) -> dict[t.LiteralString, ModelItem]:
  """List all supported models.

  \b
  ```bash
  openllm models
  ```
  """
  result: dict[t.LiteralString, ModelItem] = {
    m: ModelItem(
      architecture=config.__openllm_architecture__,
      example_id=random.choice(config.__openllm_model_ids__),
      supported_backends=config.__openllm_backend__,
      installation='pip install ' + (f'"openllm[{m}]"' if m in OPTIONAL_DEPENDENCIES or config.__openllm_requirements__ else 'openllm'),
      items=[
        str(md.tag)
        for md in bentoml.models.list()
        if 'framework' in md.info.labels
        and md.info.labels['framework'] == 'openllm'
        and 'model_name' in md.info.labels
        and md.info.labels['model_name'] == m
      ],
    )
    for m, config in CONFIG_MAPPING.items()
  }
  termui.echo(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode(), fg=None)
  return result


@cli.command()
@model_name_argument(required=False)
@click.option('-y', '--yes', '--assume-yes', is_flag=True, help='Skip confirmation when deleting a specific model')
@click.option('--include-bentos/--no-include-bentos', is_flag=True, hidden=True, default=True, help='Whether to also include pruning bentos.')
@inject
@click.pass_context
def prune_command(
  ctx: click.Context,
  model_name: str | None,
  yes: bool,
  model_store: ModelStore = Provide[BentoMLContainer.model_store],
  bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
  **_: t.Any,
) -> None:
  """Remove all saved models, and bentos built with OpenLLM locally.

  \b
  If a model type is passed, then only prune models for that given model type.
  """
  available: list[tuple[bentoml.Model | bentoml.Bento, ModelStore | BentoStore]] = [
    (m, model_store) for m in bentoml.models.list() if 'framework' in m.info.labels and m.info.labels['framework'] == 'openllm'
  ]
  if model_name is not None:
    available = [
      (m, store) for m, store in available if 'model_name' in m.info.labels and m.info.labels['model_name'] == inflection.underscore(model_name)
    ] + [
      (b, bento_store)
      for b in bentoml.bentos.list()
      if 'start_name' in b.info.labels and b.info.labels['start_name'] == inflection.underscore(model_name)
    ]
  else:
    available += [(b, bento_store) for b in bentoml.bentos.list() if '_type' in b.info.labels and '_framework' in b.info.labels]

  for store_item, store in available:
    if yes:
      delete_confirmed = True
    else:
      delete_confirmed = click.confirm(f"delete {'model' if isinstance(store, ModelStore) else 'bento'} {store_item.tag}?")
    if delete_confirmed:
      store.delete(store_item.tag)
      termui.warning(f"{store_item} deleted from {'model' if isinstance(store, ModelStore) else 'bento'} store.")
  ctx.exit(0)


def parsing_instruction_callback(
  ctx: click.Context, param: click.Parameter, value: list[str] | str | None
) -> tuple[str, bool | str] | list[str] | str | None:
  if value is None:
    return value

  if isinstance(value, list):
    # we only parse --text foo bar -> --text foo and omit bar
    value = value[-1]

  key, *values = value.split('=')
  if not key.startswith('--'):
    raise click.BadParameter(f'Invalid option format: {value}')
  key = key[2:]
  if len(values) == 0:
    return key, True
  elif len(values) == 1:
    return key, values[0]
  else:
    raise click.BadParameter(f'Invalid option format: {value}')


def shared_client_options(f: _AnyCallable | None = None) -> t.Callable[[FC], FC]:
  options = [
    click.option(
      '--endpoint',
      type=click.STRING,
      help='OpenLLM Server endpoint, i.e: http://localhost:3000',
      envvar='OPENLLM_ENDPOINT',
      show_envvar=True,
      show_default=True,
      default='http://localhost:3000',
    ),
    click.option('--timeout', type=click.INT, default=30, help='Default server timeout', show_default=True),
  ]
  return compose(*options)(f) if f is not None else compose(*options)


@cli.command()
@shared_client_options
@click.option('--server-type', type=click.Choice(['grpc', 'http']), help='Server type', default='http', show_default=True, hidden=True)
@click.option('--stream/--no-stream', type=click.BOOL, is_flag=True, default=True, help='Whether to stream the response.')
@click.argument('prompt', type=click.STRING)
@click.option(
  '--sampling-params',
  help='Define query options. (format: ``--opt temperature=0.8 --opt=top_k:12)',
  required=False,
  multiple=True,
  callback=opt_callback,
  metavar='ARG=VALUE[,ARG=VALUE]',
)
@click.pass_context
def query_command(
  ctx: click.Context,
  /,
  prompt: str,
  endpoint: str,
  timeout: int,
  stream: bool,
  server_type: t.Literal['http', 'grpc'],
  _memoized: DictStrAny,
  **_: t.Any,
) -> None:
  """Query a LLM interactively, from a terminal.

  \b
  ```bash
  $ openllm query --endpoint http://12.323.2.1:3000 "What is the meaning of life?"
  ```
  """
  if server_type == 'grpc':
    raise click.ClickException("'grpc' is currently disabled.")
  _memoized = {k: orjson.loads(v[0]) for k, v in _memoized.items() if v}
  # TODO: grpc support
  client = openllm.HTTPClient(address=endpoint, timeout=timeout)
  input_fg, generated_fg = 'magenta', 'cyan'

  if stream:
    stream_res: t.Iterator[StreamingResponse] = client.generate_stream(prompt, **_memoized)
    termui.echo(prompt, fg=input_fg, nl=False)
    for it in stream_res:
      termui.echo(it.text, fg=generated_fg, nl=False)
  else:
    termui.echo(prompt, fg=input_fg, nl=True)
    termui.echo(client.generate(prompt, **_memoized).outputs[0].text, fg=generated_fg, nl=False)
  ctx.exit(0)


@cli.group(cls=Extensions, hidden=True, name='extension')
def extension_command() -> None:
  """Extension for OpenLLM CLI."""


if __name__ == '__main__':
  cli()
