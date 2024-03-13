from __future__ import annotations

import os, logging, sys, attr, click, orjson, openllm, openllm_core, platform, typing as t
from openllm_core.utils import DEBUG_ENV_VAR, QUIET_ENV_VAR, check_bool_env, first_not_none, dantic
from openllm_core._typing_compat import Self, LiteralQuantise, LiteralSerialisation, LiteralDtype, get_literal_args
from openllm_cli import termui

logger = logging.getLogger(__name__)


@attr.define
class GlobalOptions:
  cloud_context: str | None = attr.field(default=None)

  def with_options(self, **attrs: t.Any) -> Self:
    return attr.evolve(self, **attrs)


_PACKAGE_NAME = 'openllm'


def parse_device_callback(
  _: click.Context, param: click.Parameter, value: tuple[tuple[str], ...] | None
) -> t.Tuple[str, ...] | None:
  if value is None:
    return value
  el: t.Tuple[str, ...] = tuple(i for k in value for i in k)
  # NOTE: --device all is a special case
  if len(el) == 1 and el[0] == 'all':
    return tuple(map(str, openllm.utils.available_devices()))
  return el


@click.group(context_settings=termui.CONTEXT_SETTINGS, name='openllm')
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


@cli.command(name='start', short_help='Start a LLMServer for any supported LLM.')
@click.argument('model_id', type=click.STRING, metavar='[REMOTE_REPO/MODEL_ID | /path/to/local/model]', required=True)
@click.option(
  '--revision',
  '--bentomodel-version',
  '--model-version',
  'model_version',
  type=click.STRING,
  default=None,
  help='Optional model revision to save for this model. It will be inferred automatically from model-id.',
)
@click.option(
  '--model-tag',
  '--bentomodel-tag',
  type=click.STRING,
  default=None,
  help='Optional bentomodel tag to save for this model. It will be generated automatically based on model_id and model_version if not specified.',
)
@click.option(
  '--device',
  type=dantic.CUDA,
  multiple=True,
  envvar='CUDA_VISIBLE_DEVICES',
  callback=parse_device_callback,
  help='Assign GPU devices (if available)',
  show_envvar=True,
)
@click.option('--timeout', type=int, default=360000, help='Timeout for the model executor in seconds')
@click.option(
  '--dtype',
  type=str,
  envvar='DTYPE',
  default='auto',
  help="Optional dtype for casting tensors for running inference ['float16', 'float32', 'bfloat16', 'int8', 'int16']",
)
@click.option(
  '--quantise',
  '--quantize',
  'quantize',
  type=str,
  default=None,
  envvar='QUANTIZE',
  show_envvar=True,
  help="""Dynamic quantization for running this LLM.

    The following quantization strategies are supported:

    - ``int8``: ``LLM.int8`` for [8-bit](https://arxiv.org/abs/2208.07339) quantization.

    - ``int4``: ``SpQR`` for [4-bit](https://arxiv.org/abs/2306.03078) quantization.

    - ``gptq``: ``GPTQ`` [quantization](https://arxiv.org/abs/2210.17323)

    - ``awq``: ``AWQ`` [AWQ: Activation-aware Weight Quantization](https://arxiv.org/abs/2306.00978)

    - ``squeezellm``: ``SqueezeLLM`` [SqueezeLLM: Dense-and-Sparse Quantization](https://arxiv.org/abs/2306.07629)

    > [!NOTE] that the model can also be served with quantized weights.
    """,
)
@click.option(
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
def start_command(
  model_id: str,
  model_version: str | None,
  model_tag: str | None,
  device: t.Tuple[str, ...],
  timeout: int,
  quantize: LiteralQuantise | None,
  serialisation: LiteralSerialisation | None,
  dtype: LiteralDtype | t.Literal['auto', 'float'],
  max_model_len: int | None,
  gpu_memory_utilization: float,
):
  from openllm.serialisation.transformers.weights import has_safetensors_weights
  from _bentoml_impl.server import serve_http
  from bentoml._internal.service.loader import load
  from bentoml._internal.log import configure_server_logging

  configure_server_logging()

  trust_remote_code = check_bool_env('TRUST_REMOTE_CODE', False)
  serialisation = first_not_none(
    serialisation, default='safetensors' if has_safetensors_weights(model_id, model_version) else 'legacy'
  )

  if serialisation == 'safetensors' and quantize is not None:
    logger.warning("'--quantize=%s' might not work with 'safetensors' serialisation format.", quantize)
    logger.warning(
      "Make sure to check out '%s' repository to see if the weights is in '%s' format if unsure.",
      model_id,
      serialisation,
    )
    logger.info("Tip: You can always fallback to '--serialisation legacy' when running quantisation.")

  bentomodel = openllm.prepare_model(
    model_id,
    bentomodel_tag=model_tag,
    bentomodel_version=model_version,
    quantize=quantize,
    dtype=dtype,
    serialistaion=serialisation,
    trust_remote_code=trust_remote_code,
  )
  llm_config = openllm_core.AutoConfig.from_bentomodel(bentomodel)

  # TODO: support LoRA adapters
  os.environ.update({
    QUIET_ENV_VAR: str(openllm.utils.get_quiet_mode()),
    DEBUG_ENV_VAR: str(openllm.utils.get_debug_mode()),
    'MODEL_ID': model_id,
    'SERIALIZATION': serialisation,
    'OPENLLM_CONFIG': llm_config.model_dump_json(),
    'DTYPE': dtype,
    'TRUST_REMOTE_CODE': str(trust_remote_code),
    'MAX_MODEL_LEN': orjson.dumps(max_model_len).decode(),
    'GPU_MEMORY_UTILIZATION': orjson.dumps(gpu_memory_utilization).decode(),
    'SERVICES_CONFIG': orjson.dumps(dict(traffic=dict(timeout=timeout))).decode(),
  })
  if quantize:
    os.environ['QUANTIZE'] = str(quantize)

  working_dir = os.path.abspath(os.path.dirname(__file__))
  if sys.path[0] != working_dir:
    sys.path.insert(0, working_dir)
  load('.', working_dir=working_dir).inject_config()
  serve_http('.', working_dir=working_dir)


if __name__ == '__main__':
  cli()
