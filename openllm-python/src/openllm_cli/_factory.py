from __future__ import annotations
import functools, logging, os, typing as t
import bentoml, openllm, click, inflection, click_option_group as cog
from bentoml_cli.utils import BentoMLCommandGroup
from click import shell_completion as sc

from openllm_core._configuration import LLMConfig
from openllm_core._typing_compat import Concatenate, DictStrAny, LiteralBackend, LiteralSerialisation, ParamSpec, AnyCallable, get_literal_args
from openllm_core.utils import DEBUG, compose, dantic, resolve_user_filepath


class _OpenLLM_GenericInternalConfig(LLMConfig):
  __config__ = {'name_type': 'lowercase', 'default_id': 'openllm/generic', 'model_ids': ['openllm/generic'], 'architecture': 'PreTrainedModel'}

  class GenerationConfig:
    top_k: int = 15
    top_p: float = 0.78
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
  return [sc.CompletionItem(inflection.dasherize(it), help='Model') for it in openllm.CONFIG_MAPPING if it.startswith(incomplete)]


def parse_config_options(
  config: LLMConfig, server_timeout: int, workers_per_resource: float, device: t.Tuple[str, ...] | None, cors: bool, environ: DictStrAny
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
      _bentoml_config_options_opts.extend([
        f'runners."llm-{config["start_name"]}-runner".resources."nvidia.com/gpu"[{idx}]={dev}' for idx, dev in enumerate(device)
      ])
    else:
      _bentoml_config_options_opts.append(f'runners."llm-{config["start_name"]}-runner".resources."nvidia.com/gpu"=[{device[0]}]')
  if cors:
    _bentoml_config_options_opts.extend(['api_server.http.cors.enabled=true', 'api_server.http.cors.access_control_allow_origins="*"'])
    _bentoml_config_options_opts.extend([
      f'api_server.http.cors.access_control_allow_methods[{idx}]="{it}"' for idx, it in enumerate(['GET', 'OPTIONS', 'POST', 'HEAD', 'PUT'])
    ])
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
      adapter_id = resolve_user_filepath(adapter_id, os.getcwd())
    except FileNotFoundError:
      pass
    name = adapter_name[0] if len(adapter_name) > 0 else 'default'
    ctx.params[_adapter_mapping_key][adapter_id] = name
  return None


def optimization_decorator(fn: FC, *, factory=click, _eager=True) -> FC | list[AnyCallable]:
  shared = [
    dtype_option(factory=factory),
    model_version_option(factory=factory),  #
    backend_option(factory=factory),
    quantize_option(factory=factory),  #
    serialisation_option(factory=factory),
  ]
  if not _eager:
    return shared
  return compose(*shared)(fn)


def start_decorator(fn: FC) -> FC:
  composed = compose(
    _OpenLLM_GenericInternalConfig.parse,
    parse_serve_args(),
    cog.optgroup.group(
      'LLM Options',
      help="""The following options are related to running LLM Server as well as optimization options.

          OpenLLM supports running model k-bit quantization (8-bit, 4-bit), GPTQ quantization, PagedAttention via vLLM.

          The following are either in our roadmap or currently being worked on:

          - DeepSpeed Inference: [link](https://www.deepspeed.ai/inference/)
          - GGML: Fast inference on [bare metal](https://github.com/ggerganov/ggml)
    """,
    ),
    cog.optgroup.option('--server-timeout', type=int, default=None, help='Server timeout in seconds'),
    workers_per_resource_option(factory=cog.optgroup),
    cors_option(factory=cog.optgroup),
    *optimization_decorator(fn, factory=cog.optgroup, _eager=False),
    cog.optgroup.option(
      '--device',
      type=dantic.CUDA,
      multiple=True,
      envvar='CUDA_VISIBLE_DEVICES',
      callback=parse_device_callback,
      help='Assign GPU devices (if available)',
      show_envvar=True,
    ),
    adapter_id_option(factory=cog.optgroup),
    click.option('--return-process', is_flag=True, default=False, help='Internal use only.', hidden=True),
  )

  return composed(fn)


def parse_device_callback(_: click.Context, param: click.Parameter, value: tuple[tuple[str], ...] | None) -> t.Tuple[str, ...] | None:
  if value is None:
    return value
  el: t.Tuple[str, ...] = tuple(i for k in value for i in k)
  # NOTE: --device all is a special case
  if len(el) == 1 and el[0] == 'all':
    return tuple(map(str, openllm.utils.available_devices()))
  return el


# NOTE: A list of bentoml option that is not needed for parsing.
# NOTE: User shouldn't set '--working-dir', as OpenLLM will setup this.
# NOTE: production is also deprecated
_IGNORED_OPTIONS = {'working_dir', 'production', 'protocol_version'}


def parse_serve_args() -> t.Callable[[t.Callable[..., LLMConfig]], t.Callable[[FC], FC]]:
  from bentoml_cli.cli import cli

  group = cog.optgroup.group('Start a HTTP server options', help='Related to serving the model [synonymous to `bentoml serve-http`]')

  def decorator(f: t.Callable[Concatenate[int, t.Optional[str], P], LLMConfig]) -> t.Callable[[FC], FC]:
    serve_command = cli.commands['serve']
    # The first variable is the argument bento
    # The last five is from BentoMLCommandGroup.NUMBER_OF_COMMON_PARAMS
    serve_options = [p for p in serve_command.params[1 : -BentoMLCommandGroup.NUMBER_OF_COMMON_PARAMS] if p.name not in _IGNORED_OPTIONS]
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


def adapter_id_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--adapter-id',
    default=None,
    help='Optional name or path for given LoRA adapter',
    multiple=True,
    callback=_id_callback,
    metavar='[PATH | [remote/][adapter_name:]adapter_id][, ...]',
    **attrs,
  )(f)


def cors_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--cors/--no-cors', show_default=True, default=False, envvar='OPENLLM_CORS', show_envvar=True, help='Enable CORS for the server.', **attrs
  )(f)


def machine_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option('--machine', is_flag=True, default=False, hidden=True, **attrs)(f)


def dtype_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--dtype',
    type=str,
    envvar='TORCH_DTYPE',
    default='auto',
    help="Optional dtype for casting tensors for running inference ['float16', 'float32', 'bfloat16', 'int8', 'int16']. For CTranslate2, it also accepts the following ['int8_float32', 'int8_float16', 'int8_bfloat16']",
    **attrs,
  )(f)


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


def backend_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--backend',
    type=click.Choice(get_literal_args(LiteralBackend)),
    default=None,
    envvar='OPENLLM_BACKEND',
    show_envvar=True,
    help='Runtime to use for both serialisation/inference engine.',
    **attrs,
  )(f)


def model_name_argument(f: _AnyCallable | None = None, required: bool = True, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_argument('model_name', type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING]), required=required, **attrs)(f)


def quantize_option(f: _AnyCallable | None = None, *, build: bool = False, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
    '--quantise',
    '--quantize',
    'quantize',
    type=str,
    default=None,
    envvar='OPENLLM_QUANTIZE',
    show_envvar=True,
    help="""Dynamic quantization for running this LLM.

      The following quantization strategies are supported:

      - ``int8``: ``LLM.int8`` for [8-bit](https://arxiv.org/abs/2208.07339) quantization.

      - ``int4``: ``SpQR`` for [4-bit](https://arxiv.org/abs/2306.03078) quantization.

      - ``gptq``: ``GPTQ`` [quantization](https://arxiv.org/abs/2210.17323)

      - ``awq``: ``AWQ`` [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)

      - ``squeezellm``: ``SqueezeLLM`` [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629)

      > [!NOTE] that the model can also be served with quantized weights.
      """
    + (
      """
      > [!NOTE] that this will set the mode for serving within deployment."""
      if build
      else ''
    ),
    **attrs,
  )(f)


def workers_per_resource_option(f: _AnyCallable | None = None, *, build: bool = False, **attrs: t.Any) -> t.Callable[[FC], FC]:
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

      - ``conserved``: This will determine the number of available GPU resources. For example, if ther are 4 GPUs available, then ``conserved`` is equivalent to ``--workers-per-resource 0.25``.
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
        f"'workers_per_resource' only accept '{_wpr_strategies}' as possible strategies, otherwise pass in float.", ctx, param
      ) from None
    else:
      return value
