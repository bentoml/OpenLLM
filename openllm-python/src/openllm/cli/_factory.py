from __future__ import annotations
import functools
import importlib.util
import logging
import os
import typing as t

import click
import click_option_group as cog
import inflection
import orjson

from bentoml_cli.utils import BentoMLCommandGroup
from click import shell_completion as sc
from click.shell_completion import CompletionItem

import bentoml
import openllm
import openllm_core

from bentoml._internal.configuration.containers import BentoMLContainer
from openllm_core._typing_compat import Concatenate
from openllm_core._typing_compat import DictStrAny
from openllm_core._typing_compat import LiteralBackend
from openllm_core._typing_compat import LiteralQuantise
from openllm_core._typing_compat import LiteralString
from openllm_core._typing_compat import ParamSpec
from openllm_core._typing_compat import get_literal_args
from openllm_core.utils import DEBUG

from . import termui

if t.TYPE_CHECKING:
  import subprocess

  from openllm_core._configuration import LLMConfig

logger = logging.getLogger(__name__)

P = ParamSpec('P')
LiteralOutput = t.Literal['json', 'pretty', 'porcelain']

_AnyCallable = t.Callable[..., t.Any]
FC = t.TypeVar('FC', bound=t.Union[_AnyCallable, click.Command])

def bento_complete_envvar(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[sc.CompletionItem]:
  return [sc.CompletionItem(str(it.tag), help='Bento') for it in bentoml.list() if str(it.tag).startswith(incomplete) and all(k in it.info.labels for k in {'start_name', 'bundler'})]

def model_complete_envvar(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[sc.CompletionItem]:
  return [sc.CompletionItem(inflection.dasherize(it), help='Model') for it in openllm.CONFIG_MAPPING if it.startswith(incomplete)]

def parse_config_options(config: LLMConfig, server_timeout: int, workers_per_resource: float, device: t.Tuple[str, ...] | None, cors: bool, environ: DictStrAny) -> DictStrAny:
  # TODO: Support amd.com/gpu on k8s
  _bentoml_config_options_env = environ.pop('BENTOML_CONFIG_OPTIONS', '')
  _bentoml_config_options_opts = [
      'tracing.sample_rate=1.0',
      f'api_server.traffic.timeout={server_timeout}',
      f'runners."llm-{config["start_name"]}-runner".traffic.timeout={config["timeout"]}',
      f'runners."llm-{config["start_name"]}-runner".workers_per_resource={workers_per_resource}'
  ]
  if device:
    if len(device) > 1:
      _bentoml_config_options_opts.extend([f'runners."llm-{config["start_name"]}-runner".resources."nvidia.com/gpu"[{idx}]={dev}' for idx, dev in enumerate(device)])
    else:
      _bentoml_config_options_opts.append(f'runners."llm-{config["start_name"]}-runner".resources."nvidia.com/gpu"=[{device[0]}]')
  _bentoml_config_options_opts.append(f'runners."llm-generic-embedding".resources.cpu={openllm.get_resource({"cpu":"system"},"cpu")}')
  if cors:
    _bentoml_config_options_opts.extend(['api_server.http.cors.enabled=true', 'api_server.http.cors.access_control_allow_origins="*"'])
    _bentoml_config_options_opts.extend([f'api_server.http.cors.access_control_allow_methods[{idx}]="{it}"' for idx, it in enumerate(['GET', 'OPTIONS', 'POST', 'HEAD', 'PUT'])])
  _bentoml_config_options_env += ' ' if _bentoml_config_options_env else '' + ' '.join(_bentoml_config_options_opts)
  environ['BENTOML_CONFIG_OPTIONS'] = _bentoml_config_options_env
  if DEBUG: logger.debug('Setting BENTOML_CONFIG_OPTIONS=%s', _bentoml_config_options_env)
  return environ

_adapter_mapping_key = 'adapter_map'

def _id_callback(ctx: click.Context, _: click.Parameter, value: t.Tuple[str, ...] | None) -> None:
  if not value: return None
  if _adapter_mapping_key not in ctx.params: ctx.params[_adapter_mapping_key] = {}
  for v in value:
    adapter_id, *adapter_name = v.rsplit(':', maxsplit=1)
    # try to resolve the full path if users pass in relative,
    # currently only support one level of resolve path with current directory
    try:
      adapter_id = openllm.utils.resolve_user_filepath(adapter_id, os.getcwd())
    except FileNotFoundError:
      pass
    ctx.params[_adapter_mapping_key][adapter_id] = adapter_name[0] if len(adapter_name) > 0 else None
  return None

def start_command_factory(group: click.Group, model: str, _context_settings: DictStrAny | None = None, _serve_grpc: bool = False) -> click.Command:
  llm_config = openllm.AutoConfig.for_model(model)
  command_attrs: DictStrAny = dict(name=llm_config['model_name'],
                                   context_settings=_context_settings or termui.CONTEXT_SETTINGS,
                                   short_help=f"Start a LLMServer for '{model}'",
                                   aliases=[llm_config['start_name']] if llm_config['name_type'] == 'dasherize' else None,
                                   help=f'''\
{llm_config['env'].start_docstring}

\b
Note: ``{llm_config['start_name']}`` can also be run with any other models available on HuggingFace
or fine-tuned variants as long as it belongs to the architecture generation ``{llm_config['architecture']}`` (trust_remote_code={llm_config['trust_remote_code']}).

\b
For example: One can start [Fastchat-T5](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) with ``openllm start flan-t5``:

\b
$ openllm start flan-t5 --model-id lmsys/fastchat-t5-3b-v1.0

\b
Available official model_id(s): [default: {llm_config['default_id']}]

\b
{orjson.dumps(llm_config['model_ids'], option=orjson.OPT_INDENT_2).decode()}
''',
                                   )

  if llm_config['requires_gpu'] and openllm.utils.device_count() < 1:
    # NOTE: The model requires GPU, therefore we will return a dummy command
    command_attrs.update({
        'short_help': '(Disabled because there is no GPU available)', 'help': f'{model} is currently not available to run on your local machine because it requires GPU for inference.'
    })
    return noop_command(group, llm_config, _serve_grpc, **command_attrs)

  @group.command(**command_attrs)
  @start_decorator(llm_config, serve_grpc=_serve_grpc)
  @click.pass_context
  def start_cmd(ctx: click.Context,
                /,
                server_timeout: int,
                model_id: str | None,
                model_version: str | None,
                workers_per_resource: t.Literal['conserved', 'round_robin'] | LiteralString,
                device: t.Tuple[str, ...],
                quantize: LiteralQuantise | None,
                backend: LiteralBackend,
                serialisation: t.Literal['safetensors', 'legacy'],
                cors: bool,
                adapter_id: str | None,
                return_process: bool,
                **attrs: t.Any,
                ) -> LLMConfig | subprocess.Popen[bytes]:
    if serialisation == 'safetensors' and quantize is not None and openllm_core.utils.check_bool_env('OPENLLM_SERIALIZATION_WARNING'):
      termui.echo(
          f"'--quantize={quantize}' might not work with 'safetensors' serialisation format. Use with caution!. To silence this warning, set \"OPENLLM_SERIALIZATION_WARNING=False\"\nNote: You can always fallback to '--serialisation legacy' when running quantisation.",
          fg='yellow')
    adapter_map: dict[str, str | None] | None = attrs.pop(_adapter_mapping_key, None)
    config, server_attrs = llm_config.model_validate_click(**attrs)
    server_timeout = openllm.utils.first_not_none(server_timeout, default=config['timeout'])
    server_attrs.update({'working_dir': os.path.dirname(os.path.dirname(__file__)), 'timeout': server_timeout})
    if _serve_grpc: server_attrs['grpc_protocol_version'] = 'v1'
    # NOTE: currently, theres no development args in bentoml.Server. To be fixed upstream.
    development = server_attrs.pop('development')
    server_attrs.setdefault('production', not development)
    wpr = openllm.utils.first_not_none(workers_per_resource, default=config['workers_per_resource'])

    if isinstance(wpr, str):
      if wpr == 'round_robin': wpr = 1.0
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

    # Create a new model env to work with the envvar during CLI invocation
    env = openllm.utils.EnvVarMixin(config['model_name'], backend, model_id=model_id or config['default_id'], quantize=quantize)
    requirements = llm_config['requirements']
    if requirements is not None and len(requirements) > 0:
      missing_requirements = [i for i in requirements if importlib.util.find_spec(inflection.underscore(i)) is None]
      if len(missing_requirements) > 0:
        termui.echo(f'Make sure to have the following dependencies available: {missing_requirements}', fg='yellow')

    # NOTE: This is to set current configuration
    start_env = os.environ.copy()
    start_env = parse_config_options(config, server_timeout, wpr, device, cors, start_env)

    start_env.update({
        'OPENLLM_MODEL': model,
        'BENTOML_DEBUG': str(openllm.utils.get_debug_mode()),
        'BENTOML_HOME': os.environ.get('BENTOML_HOME', BentoMLContainer.bentoml_home.get()),
        'OPENLLM_ADAPTER_MAP': orjson.dumps(adapter_map).decode(),
        'OPENLLM_SERIALIZATION': serialisation,
        env.backend: env['backend_value'],
    })
    if env['model_id_value']: start_env[env.model_id] = str(env['model_id_value'])
    if env['quantize_value']: start_env[env.quantize] = str(env['quantize_value'])

    llm = openllm.utils.infer_auto_class(env['backend_value']).for_model(model,
                                                                         model_id=start_env[env.model_id],
                                                                         model_version=model_version,
                                                                         llm_config=config,
                                                                         ensure_available=True,
                                                                         adapter_map=adapter_map,
                                                                         quantize=env['quantize_value'],
                                                                         serialisation=serialisation)
    start_env.update({env.config: llm.config.model_dump_json().decode()})

    server = bentoml.GrpcServer('_service:svc', **server_attrs) if _serve_grpc else bentoml.HTTPServer('_service:svc', **server_attrs)
    openllm.utils.analytics.track_start_init(llm.config)

    def next_step(model_name: str, adapter_map: DictStrAny | None) -> None:
      cmd_name = f'openllm build {model_name}'
      if adapter_map is not None:
        cmd_name += ' ' + ' '.join([f'--adapter-id {s}' for s in [f'{p}:{name}' if name not in (None, 'default') else p for p, name in adapter_map.items()]])
      if not openllm.utils.get_quiet_mode():
        termui.echo(f"\n🚀 Next step: run '{cmd_name}' to create a Bento for {model_name}", fg='blue')

    if return_process:
      server.start(env=start_env, text=True)
      if server.process is None: raise click.ClickException('Failed to start the server.')
      return server.process
    else:
      try:
        server.start(env=start_env, text=True, blocking=True)
      except KeyboardInterrupt:
        next_step(model, adapter_map)
      except Exception as err:
        termui.echo(f'Error caught while running LLM Server:\n{err}', fg='red')
      else:
        next_step(model, adapter_map)

    # NOTE: Return the configuration for telemetry purposes.
    return config

  return start_cmd

def noop_command(group: click.Group, llm_config: LLMConfig, _serve_grpc: bool, **command_attrs: t.Any) -> click.Command:
  context_settings = command_attrs.pop('context_settings', {})
  context_settings.update({'ignore_unknown_options': True, 'allow_extra_args': True})
  command_attrs['context_settings'] = context_settings
  # NOTE: The model requires GPU, therefore we will return a dummy command
  @group.command(**command_attrs)
  def noop(**_: t.Any) -> LLMConfig:
    termui.echo('No GPU available, therefore this command is disabled', fg='red')
    openllm.utils.analytics.track_start_init(llm_config)
    return llm_config

  return noop

def start_decorator(llm_config: LLMConfig, serve_grpc: bool = False) -> t.Callable[[FC], t.Callable[[FC], FC]]:
  def wrapper(fn: FC) -> t.Callable[[FC], FC]:
    composed = openllm.utils.compose(
        llm_config.to_click_options,
        _http_server_args if not serve_grpc else _grpc_server_args,
        cog.optgroup.group('General LLM Options', help=f"The following options are related to running '{llm_config['start_name']}' LLM Server."),
        model_id_option(factory=cog.optgroup),
        model_version_option(factory=cog.optgroup),
        cog.optgroup.option('--server-timeout', type=int, default=None, help='Server timeout in seconds'),
        workers_per_resource_option(factory=cog.optgroup),
        cors_option(factory=cog.optgroup),
        backend_option(factory=cog.optgroup),
        cog.optgroup.group('LLM Optimization Options',
                           help='''Optimization related options.

            OpenLLM supports running model k-bit quantization (8-bit, 4-bit), GPTQ quantization, PagedAttention via vLLM.

            The following are either in our roadmap or currently being worked on:

            - DeepSpeed Inference: [link](https://www.deepspeed.ai/inference/)
            - GGML: Fast inference on [bare metal](https://github.com/ggerganov/ggml)
            '''),
        quantize_option(factory=cog.optgroup),
        serialisation_option(factory=cog.optgroup),
        cog.optgroup.option('--device',
                            type=openllm.utils.dantic.CUDA,
                            multiple=True,
                            envvar='CUDA_VISIBLE_DEVICES',
                            callback=parse_device_callback,
                            help=f"Assign GPU devices (if available) for {llm_config['model_name']}.",
                            show_envvar=True),
        cog.optgroup.group('Fine-tuning related options',
                           help='''\
    Note that the argument `--adapter-id` can accept the following format:

    - `--adapter-id /path/to/adapter` (local adapter)

    - `--adapter-id remote/adapter` (remote adapter from HuggingFace Hub)

    - `--adapter-id remote/adapter:eng_lora` (two previous adapter options with the given adapter_name)

    ```bash

    $ openllm start opt --adapter-id /path/to/adapter_dir --adapter-id remote/adapter:eng_lora

    ```
    '''),
        cog.optgroup.option('--adapter-id',
                            default=None,
                            help='Optional name or path for given LoRA adapter' + f" to wrap '{llm_config['model_name']}'",
                            multiple=True,
                            callback=_id_callback,
                            metavar='[PATH | [remote/][adapter_name:]adapter_id][, ...]'),
        click.option('--return-process', is_flag=True, default=False, help='Internal use only.', hidden=True),
    )
    return composed(fn)

  return wrapper

def parse_device_callback(ctx: click.Context, param: click.Parameter, value: tuple[tuple[str], ...] | None) -> t.Tuple[str, ...] | None:
  if value is None: return value
  if not isinstance(value, tuple): ctx.fail(f'{param} only accept multiple values, not {type(value)} (value: {value})')
  el: t.Tuple[str, ...] = tuple(i for k in value for i in k)
  # NOTE: --device all is a special case
  if len(el) == 1 and el[0] == 'all': return tuple(map(str, openllm.utils.available_devices()))
  return el

# NOTE: A list of bentoml option that is not needed for parsing.
# NOTE: User shouldn't set '--working-dir', as OpenLLM will setup this.
# NOTE: production is also deprecated
_IGNORED_OPTIONS = {'working_dir', 'production', 'protocol_version'}

def parse_serve_args(serve_grpc: bool) -> t.Callable[[t.Callable[..., LLMConfig]], t.Callable[[FC], FC]]:
  '''Parsing `bentoml serve|serve-grpc` click.Option to be parsed via `openllm start`.'''
  from bentoml_cli.cli import cli

  command = 'serve' if not serve_grpc else 'serve-grpc'
  group = cog.optgroup.group(f"Start a {'HTTP' if not serve_grpc else 'gRPC'} server options",
                             help=f"Related to serving the model [synonymous to `bentoml {'serve-http' if not serve_grpc else command }`]",
                             )

  def decorator(f: t.Callable[Concatenate[int, t.Optional[str], P], LLMConfig]) -> t.Callable[[FC], FC]:
    serve_command = cli.commands[command]
    # The first variable is the argument bento
    # The last five is from BentoMLCommandGroup.NUMBER_OF_COMMON_PARAMS
    serve_options = [p for p in serve_command.params[1:-BentoMLCommandGroup.NUMBER_OF_COMMON_PARAMS] if p.name not in _IGNORED_OPTIONS]
    for options in reversed(serve_options):
      attrs = options.to_info_dict()
      # we don't need param_type_name, since it should all be options
      attrs.pop('param_type_name')
      # name is not a valid args
      attrs.pop('name')
      # type can be determine from default value
      attrs.pop('type')
      param_decls = (*attrs.pop('opts'), *attrs.pop('secondary_opts'))
      f = cog.optgroup.option(*param_decls, **attrs)(f)
    return group(f)

  return decorator

_http_server_args, _grpc_server_args = parse_serve_args(False), parse_serve_args(True)

def _click_factory_type(*param_decls: t.Any, **attrs: t.Any) -> t.Callable[[FC | None], FC]:
  '''General ``@click`` decorator with some sauce.

  This decorator extends the default ``@click.option`` plus a factory option and factory attr to
  provide type-safe click.option or click.argument wrapper for all compatible factory.
  '''
  factory = attrs.pop('factory', click)
  factory_attr = attrs.pop('attr', 'option')
  if factory_attr != 'argument': attrs.setdefault('help', 'General option for OpenLLM CLI.')

  def decorator(f: FC | None) -> FC:
    callback = getattr(factory, factory_attr, None)
    if callback is None: raise ValueError(f'Factory {factory} has no attribute {factory_attr}.')
    return t.cast(FC, callback(*param_decls, **attrs)(f) if f is not None else callback(*param_decls, **attrs))

  return decorator

cli_option = functools.partial(_click_factory_type, attr='option')
cli_argument = functools.partial(_click_factory_type, attr='argument')

def output_option(f: _AnyCallable | None = None, *, default_value: LiteralOutput = 'pretty', **attrs: t.Any) -> t.Callable[[FC], FC]:
  output = ['json', 'pretty', 'porcelain']

  def complete_output_var(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
    return [CompletionItem(it) for it in output]

  return cli_option('-o',
                    '--output',
                    'output',
                    type=click.Choice(output),
                    default=default_value,
                    help='Showing output type.',
                    show_default=True,
                    envvar='OPENLLM_OUTPUT',
                    show_envvar=True,
                    shell_complete=complete_output_var,
                    **attrs)(f)

def cors_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option('--cors/--no-cors', show_default=True, default=False, envvar='OPENLLM_CORS', show_envvar=True, help='Enable CORS for the server.', **attrs)(f)

def machine_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option('--machine', is_flag=True, default=False, hidden=True, **attrs)(f)

def model_id_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option('--model-id', type=click.STRING, default=None, envvar='OPENLLM_MODEL_ID', show_envvar=True, help='Optional model_id name or path for (fine-tune) weight.', **attrs)(f)

def model_version_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option('--model-version', type=click.STRING, default=None, help='Optional model version to save for this model. It will be inferred automatically from model-id.', **attrs)(f)

def backend_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  # NOTE: LiteralBackend needs to remove the last two item as ggml and mlc is wip
  # XXX: remove the check for __args__ once we have ggml and mlc supports
  return cli_option('--backend',
                    type=click.Choice(get_literal_args(LiteralBackend)[:-2]),
                    default='pt',
                    envvar='OPENLLM_BACKEND',
                    show_envvar=True,
                    help='The implementation for saving this LLM.',
                    **attrs)(f)

def model_name_argument(f: _AnyCallable | None = None, required: bool = True, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_argument('model_name', type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING]), required=required, **attrs)(f)

def quantize_option(f: _AnyCallable | None = None, *, build: bool = False, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option('--quantise',
                    '--quantize',
                    'quantize',
                    type=click.Choice(['int8', 'int4', 'gptq']),
                    default=None,
                    envvar='OPENLLM_QUANTIZE',
                    show_envvar=True,
                    help='''Dynamic quantization for running this LLM.

      The following quantization strategies are supported:

      - ``int8``: ``LLM.int8`` for [8-bit](https://arxiv.org/abs/2208.07339) quantization.

      - ``int4``: ``SpQR`` for [4-bit](https://arxiv.org/abs/2306.03078) quantization.

      - ``gptq``: ``GPTQ`` [quantization](https://arxiv.org/abs/2210.17323)

      > [!NOTE] that the model can also be served with quantized weights.
      ''' + ('''
      > [!NOTE] that this will set the mode for serving within deployment.''' if build else '') + '''
      > [!NOTE] that quantization are currently only available in *PyTorch* models.''',
                    **attrs)(f)

def workers_per_resource_option(f: _AnyCallable | None = None, *, build: bool = False, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option('--workers-per-resource',
                    default=None,
                    callback=workers_per_resource_callback,
                    type=str,
                    required=False,
                    help='''Number of workers per resource assigned.

      See https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy
      for more information. By default, this is set to 1.

      > [!NOTE] ``--workers-per-resource`` will also accept the following strategies:

      - ``round_robin``: Similar behaviour when setting ``--workers-per-resource 1``. This is useful for smaller models.

      - ``conserved``: This will determine the number of available GPU resources, and only assign one worker for the LLMRunner. For example, if ther are 4 GPUs available, then ``conserved`` is equivalent to ``--workers-per-resource 0.25``.
      ''' + ("""\n
      > [!NOTE] The workers value passed into 'build' will determine how the LLM can
      > be provisioned in Kubernetes as well as in standalone container. This will
      > ensure it has the same effect with 'openllm start --api-workers ...'""" if build else ''),
                    **attrs)(f)

def serialisation_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option('--serialisation',
                    '--serialization',
                    'serialisation',
                    type=click.Choice(['safetensors', 'legacy']),
                    default='safetensors',
                    show_default=True,
                    show_envvar=True,
                    envvar='OPENLLM_SERIALIZATION',
                    help='''Serialisation format for save/load LLM.

      Currently the following strategies are supported:

      - ``safetensors``: This will use safetensors format, which is synonymous to

                  \b
                  ``safe_serialization=True``.

                  \b
                  > [!NOTE] that this format might not work for every cases, and
                  you can always fallback to ``legacy`` if needed.

      - ``legacy``: This will use PyTorch serialisation format, often as ``.bin`` files. This should be used if the model doesn't yet support safetensors.

      > [!NOTE] that GGML format is working in progress.
      ''',
                    **attrs)(f)

def container_registry_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option('--container-registry',
                    'container_registry',
                    type=click.Choice(list(openllm.bundle.CONTAINER_NAMES)),
                    default='ecr',
                    show_default=True,
                    show_envvar=True,
                    envvar='OPENLLM_CONTAINER_REGISTRY',
                    callback=container_registry_callback,
                    help='The default container registry to get the base image for building BentoLLM. Currently, it supports ecr, ghcr, docker',
                    **attrs)(f)

_wpr_strategies = {'round_robin', 'conserved'}

def workers_per_resource_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
  if value is None: return value
  value = inflection.underscore(value)
  if value in _wpr_strategies: return value
  else:
    try:
      float(value)  # type: ignore[arg-type]
    except ValueError:
      raise click.BadParameter(f"'workers_per_resource' only accept '{_wpr_strategies}' as possible strategies, otherwise pass in float.", ctx, param) from None
    else:
      return value

def container_registry_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
  if value is None: return value
  if value not in openllm.bundle.supported_registries:
    raise click.BadParameter(f'Value must be one of {openllm.bundle.supported_registries}', ctx, param)
  return value
