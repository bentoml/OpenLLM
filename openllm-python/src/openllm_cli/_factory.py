from __future__ import annotations
import functools
import logging
import os
import typing as t

import click
import click_option_group as cog
import inflection

from bentoml_cli.utils import BentoMLCommandGroup
from click import ClickException
from click import shell_completion as sc

import bentoml
import openllm

from openllm_core._configuration import LLMConfig
from openllm_core._typing_compat import Concatenate
from openllm_core._typing_compat import DictStrAny
from openllm_core._typing_compat import LiteralBackend
from openllm_core._typing_compat import LiteralQuantise
from openllm_core._typing_compat import LiteralSerialisation
from openllm_core._typing_compat import ParamSpec
from openllm_core._typing_compat import get_literal_args
from openllm_core.utils import DEBUG


class _OpenLLM_GenericInternalConfig(LLMConfig):
  __config__ = {
    'name_type': 'lowercase',
    'default_id': 'openllm/generic',
    'model_ids': ['openllm/generic'],
    'architecture': 'PreTrainedModel',
  }

  class GenerationConfig:
    top_k: int = 15
    top_p: float = 0.9
    temperature: float = 0.75
    max_new_tokens: int = 128


logger = logging.getLogger(__name__)

P = ParamSpec('P')
LiteralOutput = t.Literal['json', 'pretty', 'porcelain']

_AnyCallable = t.Callable[..., t.Any]
FC = t.TypeVar('FC', bound=t.Union[_AnyCallable, click.Command])


def bento_complete_envvar(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[sc.CompletionItem]:
  return [
    sc.CompletionItem(str(it.tag), help='Bento')
    for it in bentoml.list()
    if str(it.tag).startswith(incomplete) and all(k in it.info.labels for k in {'start_name', 'bundler'})
  ]


def model_complete_envvar(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[sc.CompletionItem]:
  return [
    sc.CompletionItem(inflection.dasherize(it), help='Model')
    for it in openllm.CONFIG_MAPPING
    if it.startswith(incomplete)
  ]


def parse_config_options(
  config: LLMConfig,
  server_timeout: int,
  workers_per_resource: float,
  device: t.Tuple[str, ...] | None,
  cors: bool,
  environ: DictStrAny,
) -> DictStrAny:
  # TODO: Support amd.com/gpu on k8s
  _bentoml_config_options_env = environ.pop('BENTOML_CONFIG_OPTIONS', '')
  _bentoml_config_options_opts = [
    'tracing.sample_rate=1.0',
    'api_server.max_runner_connections=25',
    f'runners."llm-{config["start_name"]}-runner".batching.max_batch_size=128',
    f'api_server.traffic.timeout={server_timeout}',
    f'runners."llm-{config["start_name"]}-runner".traffic.timeout={config["timeout"]}',
    f'runners."llm-{config["start_name"]}-runner".workers_per_resource={workers_per_resource}',
  ]
  if device:
    if len(device) > 1:
      _bentoml_config_options_opts.extend(
        [
          f'runners."llm-{config["start_name"]}-runner".resources."nvidia.com/gpu"[{idx}]={dev}'
          for idx, dev in enumerate(device)
        ]
      )
    else:
      _bentoml_config_options_opts.append(
        f'runners."llm-{config["start_name"]}-runner".resources."nvidia.com/gpu"=[{device[0]}]'
      )
  if cors:
    _bentoml_config_options_opts.extend(
      ['api_server.http.cors.enabled=true', 'api_server.http.cors.access_control_allow_origins="*"']
    )
    _bentoml_config_options_opts.extend(
      [
        f'api_server.http.cors.access_control_allow_methods[{idx}]="{it}"'
        for idx, it in enumerate(['GET', 'OPTIONS', 'POST', 'HEAD', 'PUT'])
      ]
    )
  _bentoml_config_options_env += ' ' if _bentoml_config_options_env else '' + ' '.join(_bentoml_config_options_opts)
  environ['BENTOML_CONFIG_OPTIONS'] = _bentoml_config_options_env
  if DEBUG:
    logger.debug('Setting BENTOML_CONFIG_OPTIONS=%s', _bentoml_config_options_env)
  return environ


_adapter_mapping_key = 'adapter_map'


def _id_callback(ctx: click.Context, _: click.Parameter, value: t.Tuple[str, ...] | None) -> None:
  if not value:
    return None
  if _adapter_mapping_key not in ctx.params:
    ctx.params[_adapter_mapping_key] = {}
  for v in value:
    adapter_id, *adapter_name = v.rsplit(':', maxsplit=1)
    # try to resolve the full path if users pass in relative,
    # currently only support one level of resolve path with current directory
    try:
      adapter_id = openllm.utils.resolve_user_filepath(adapter_id, os.getcwd())
    except FileNotFoundError:
      pass
    if len(adapter_name) == 0:
      raise ClickException(f'Adapter name is required for {adapter_id}')
    ctx.params[_adapter_mapping_key][adapter_id] = adapter_name[0]
  return None


def start_decorator(serve_grpc: bool = False) -> t.Callable[[FC], t.Callable[[FC], FC]]:
  def wrapper(fn: FC) -> t.Callable[[FC], FC]:
    composed = openllm.utils.compose(
      _OpenLLM_GenericInternalConfig().to_click_options,
      _http_server_args if not serve_grpc else _grpc_server_args,
      cog.optgroup.group('General LLM Options', help='The following options are related to running LLM Server.'),
      model_version_option(factory=cog.optgroup),
      system_message_option(factory=cog.optgroup),
      prompt_template_file_option(factory=cog.optgroup),
      cog.optgroup.option('--server-timeout', type=int, default=None, help='Server timeout in seconds'),
      workers_per_resource_option(factory=cog.optgroup),
      cors_option(factory=cog.optgroup),
      backend_option(factory=cog.optgroup),
      cog.optgroup.group(
        'LLM Optimization Options',
        help="""Optimization related options.

            OpenLLM supports running model k-bit quantization (8-bit, 4-bit), GPTQ quantization, PagedAttention via vLLM.

            The following are either in our roadmap or currently being worked on:

            - DeepSpeed Inference: [link](https://www.deepspeed.ai/inference/)
            - GGML: Fast inference on [bare metal](https://github.com/ggerganov/ggml)
            """,
      ),
      quantize_option(factory=cog.optgroup),
      serialisation_option(factory=cog.optgroup),
      cog.optgroup.option(
        '--device',
        type=openllm.utils.dantic.CUDA,
        multiple=True,
        envvar='CUDA_VISIBLE_DEVICES',
        callback=parse_device_callback,
        help='Assign GPU devices (if available)',
        show_envvar=True,
      ),
      cog.optgroup.group(
        'Fine-tuning related options',
        help="""\
    Note that the argument `--adapter-id` can accept the following format:

    - `--adapter-id /path/to/adapter` (local adapter)

    - `--adapter-id remote/adapter` (remote adapter from HuggingFace Hub)

    - `--adapter-id remote/adapter:eng_lora` (two previous adapter options with the given adapter_name)

    ```bash

    $ openllm start opt --adapter-id /path/to/adapter_dir --adapter-id remote/adapter:eng_lora

    ```
    """,
      ),
      cog.optgroup.option(
        '--adapter-id',
        default=None,
        help='Optional name or path for given LoRA adapter',
        multiple=True,
        callback=_id_callback,
        metavar='[PATH | [remote/][adapter_name:]adapter_id][, ...]',
      ),
      click.option('--return-process', is_flag=True, default=False, help='Internal use only.', hidden=True),
    )
    return composed(fn)

  return wrapper


def parse_device_callback(
  ctx: click.Context, param: click.Parameter, value: tuple[tuple[str], ...] | None
) -> t.Tuple[str, ...] | None:
  if value is None:
    return value
  if not isinstance(value, tuple):
    ctx.fail(f'{param} only accept multiple values, not {type(value)} (value: {value})')
  el: t.Tuple[str, ...] = tuple(i for k in value for i in k)
  # NOTE: --device all is a special case
  if len(el) == 1 and el[0] == 'all':
    return tuple(map(str, openllm.utils.available_devices()))
  return el


# NOTE: A list of bentoml option that is not needed for parsing.
# NOTE: User shouldn't set '--working-dir', as OpenLLM will setup this.
# NOTE: production is also deprecated
_IGNORED_OPTIONS = {'working_dir', 'production', 'protocol_version'}


def parse_serve_args(serve_grpc: bool) -> t.Callable[[t.Callable[..., LLMConfig]], t.Callable[[FC], FC]]:
  """Parsing `bentoml serve|serve-grpc` click.Option to be parsed via `openllm start`."""
  from bentoml_cli.cli import cli

  command = 'serve' if not serve_grpc else 'serve-grpc'
  group = cog.optgroup.group(
    f"Start a {'HTTP' if not serve_grpc else 'gRPC'} server options",
    help=f"Related to serving the model [synonymous to `bentoml {'serve-http' if not serve_grpc else command }`]",
  )

  def decorator(f: t.Callable[Concatenate[int, t.Optional[str], P], LLMConfig]) -> t.Callable[[FC], FC]:
    serve_command = cli.commands[command]
    # The first variable is the argument bento
    # The last five is from BentoMLCommandGroup.NUMBER_OF_COMMON_PARAMS
    serve_options = [
      p
      for p in serve_command.params[1 : -BentoMLCommandGroup.NUMBER_OF_COMMON_PARAMS]
      if p.name not in _IGNORED_OPTIONS
    ]
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
  """General ``@click`` decorator with some sauce.

  This decorator extends the default ``@click.option`` plus a factory option and factory attr to
  provide type-safe click.option or click.argument wrapper for all compatible factory.
  """
  factory = attrs.pop('factory', click)
  factory_attr = attrs.pop('attr', 'option')
  if factory_attr != 'argument':
    attrs.setdefault('help', 'General option for OpenLLM CLI.')

  def decorator(f: FC | None) -> FC:
    callback = getattr(factory, factory_attr, None)
    if callback is None:
      raise ValueError(f'Factory {factory} has no attribute {factory_attr}.')
    return t.cast(FC, callback(*param_decls, **attrs)(f) if f is not None else callback(*param_decls, **attrs))

  return decorator


cli_option = functools.partial(_click_factory_type, attr='option')
cli_argument = functools.partial(_click_factory_type, attr='argument')


def cors_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--cors/--no-cors',
    show_default=True,
    default=False,
    envvar='OPENLLM_CORS',
    show_envvar=True,
    help='Enable CORS for the server.',
    **attrs,
  )(f)


def machine_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option('--machine', is_flag=True, default=False, hidden=True, **attrs)(f)


def model_id_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--model-id',
    type=click.STRING,
    default=None,
    envvar='OPENLLM_MODEL_ID',
    show_envvar=True,
    help='Optional model_id name or path for (fine-tune) weight.',
    **attrs,
  )(f)


def model_version_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--model-version',
    type=click.STRING,
    default=None,
    help='Optional model version to save for this model. It will be inferred automatically from model-id.',
    **attrs,
  )(f)


def system_message_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--system-message',
    type=click.STRING,
    default=None,
    envvar='OPENLLM_SYSTEM_MESSAGE',
    help='Optional system message for supported LLMs. If given LLM supports system message, OpenLLM will provide a default system message.',
    **attrs,
  )(f)


def prompt_template_file_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--prompt-template-file',
    type=click.File(),
    default=None,
    help='Optional file path containing user-defined custom prompt template. By default, the prompt template for the specified LLM will be used.',
    **attrs,
  )(f)


def backend_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  # NOTE: LiteralBackend needs to remove the last two item as ggml and mlc is wip
  # XXX: remove the check for __args__ once we have ggml and mlc supports
  return cli_option(
    '--backend',
    type=click.Choice(get_literal_args(LiteralBackend)[:2]),
    default=None,
    envvar='OPENLLM_BACKEND',
    show_envvar=True,
    help='The implementation for saving this LLM.',
    **attrs,
  )(f)


def model_name_argument(f: _AnyCallable | None = None, required: bool = True, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_argument(
    'model_name',
    type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING]),
    required=required,
    **attrs,
  )(f)


def quantize_option(f: _AnyCallable | None = None, *, build: bool = False, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--quantise',
    '--quantize',
    'quantize',
    type=click.Choice(get_literal_args(LiteralQuantise)),
    default=None,
    envvar='OPENLLM_QUANTIZE',
    show_envvar=True,
    help="""Dynamic quantization for running this LLM.

      The following quantization strategies are supported:

      - ``int8``: ``LLM.int8`` for [8-bit](https://arxiv.org/abs/2208.07339) quantization.

      - ``int4``: ``SpQR`` for [4-bit](https://arxiv.org/abs/2306.03078) quantization.

      - ``gptq``: ``GPTQ`` [quantization](https://arxiv.org/abs/2210.17323)

      > [!NOTE] that the model can also be served with quantized weights.
      """
    + (
      """
      > [!NOTE] that this will set the mode for serving within deployment."""
      if build
      else ''
    )
    + """
      > [!NOTE] that quantization are currently only available in *PyTorch* models.""",
    **attrs,
  )(f)


def workers_per_resource_option(
  f: _AnyCallable | None = None, *, build: bool = False, **attrs: t.Any
) -> t.Callable[[FC], FC]:
  return cli_option(
    '--workers-per-resource',
    default=None,
    callback=workers_per_resource_callback,
    type=str,
    required=False,
    help="""Number of workers per resource assigned.

      See https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy
      for more information. By default, this is set to 1.

      > [!NOTE] ``--workers-per-resource`` will also accept the following strategies:

      - ``round_robin``: Similar behaviour when setting ``--workers-per-resource 1``. This is useful for smaller models.

      - ``conserved``: This will determine the number of available GPU resources, and only assign one worker for the LLMRunner. For example, if ther are 4 GPUs available, then ``conserved`` is equivalent to ``--workers-per-resource 0.25``.
      """
    + (
      """\n
      > [!NOTE] The workers value passed into 'build' will determine how the LLM can
      > be provisioned in Kubernetes as well as in standalone container. This will
      > ensure it has the same effect with 'openllm start --api-workers ...'"""
      if build
      else ''
    ),
    **attrs,
  )(f)


def serialisation_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--serialisation',
    '--serialization',
    'serialisation',
    type=click.Choice(get_literal_args(LiteralSerialisation)),
    default=None,
    show_default=True,
    show_envvar=True,
    envvar='OPENLLM_SERIALIZATION',
    help="""Serialisation format for save/load LLM.

      Currently the following strategies are supported:

      - ``safetensors``: This will use safetensors format, which is synonymous to ``safe_serialization=True``.

      > [!NOTE] Safetensors might not work for every cases, and you can always fallback to ``legacy`` if needed.

      - ``legacy``: This will use PyTorch serialisation format, often as ``.bin`` files. This should be used if the model doesn't yet support safetensors.
      """,
    **attrs,
  )(f)


def container_registry_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--container-registry',
    'container_registry',
    type=click.Choice(list(openllm.bundle.CONTAINER_NAMES)),
    default='ecr',
    show_default=True,
    show_envvar=True,
    envvar='OPENLLM_CONTAINER_REGISTRY',
    callback=container_registry_callback,
    help='The default container registry to get the base image for building BentoLLM. Currently, it supports ecr, ghcr, docker',
    **attrs,
  )(f)


_wpr_strategies = {'round_robin', 'conserved'}


def workers_per_resource_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
  if value is None:
    return value
  value = inflection.underscore(value)
  if value in _wpr_strategies:
    return value
  else:
    try:
      float(value)  # type: ignore[arg-type]
    except ValueError:
      raise click.BadParameter(
        f"'workers_per_resource' only accept '{_wpr_strategies}' as possible strategies, otherwise pass in float.",
        ctx,
        param,
      ) from None
    else:
      return value


def container_registry_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
  if value is None:
    return value
  if value not in openllm.bundle.supported_registries:
    raise click.BadParameter(f'Value must be one of {openllm.bundle.supported_registries}', ctx, param)
  return value
