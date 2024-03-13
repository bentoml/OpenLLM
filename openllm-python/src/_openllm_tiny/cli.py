from __future__ import annotations

import os, logging, attr, click, orjson, openllm, platform
import sys
import subprocess
from openllm_core.utils import check_bool_env, first_not_none
import typing as t
from bentoml._internal.configuration.containers import BentoMLContainer
from openllm_core._typing_compat import Self, LiteralQuantise, LiteralSerialisation, LiteralDtype
from openllm_cli import termui
from openllm_cli._factory import start_decorator

logger = logging.getLogger(__name__)

OPENLLM_FIGLET = """\
 ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝
"""


@attr.define
class GlobalOptions:
  cloud_context: str | None = attr.field(default=None)

  def with_options(self, **attrs: t.Any) -> Self:
    return attr.evolve(self, **attrs)


_PACKAGE_NAME = 'openllm'


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
  model_version: str | None,
  device: t.Tuple[str, ...],
  quantize: LiteralQuantise | None,
  serialisation: LiteralSerialisation | None,
  adapter_id: str | None,
  dtype: LiteralDtype | t.Literal['auto', 'float'],
  max_model_len: int | None,
  gpu_memory_utilization: float,
  **attrs: t.Any,
):
  adapter_map: dict[str, str] | None = attrs.pop('adapter_map', None)

  from openllm.serialisation.transformers.weights import has_safetensors_weights

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

  trust_remote_code = check_bool_env('TRUST_REMOTE_CODE', False)

  bentomodel = openllm.prepare_model(
    model_id,
    bentomodel_version=model_version,
    quantize=quantize,
    dtype=dtype,
    serialistaion=serialisation,
    trust_remote_code=trust_remote_code,
  )
  llm_config = openllm.AutoConfig.from_bentomodel(bentomodel)

  environ = os.environ.copy()
  environ.update({
    'OPENLLM_MODEL_ID': model_id,
    'BENTOML_DEBUG': str(openllm.utils.get_debug_mode()),
    'BENTOML_HOME': os.environ.get('BENTOML_HOME', BentoMLContainer.bentoml_home.get()),
    'OPENLLM_ADAPTER_MAP': orjson.dumps(adapter_map).decode(),
    'OPENLLM_SERIALIZATION': serialisation,
    'OPENLLM_CONFIG': llm_config.model_dump_json(),
    'DTYPE': dtype,
    'TRUST_REMOTE_CODE': str(trust_remote_code),
    'MAX_MODEL_LEN': orjson.dumps(max_model_len).decode(),
    'GPU_MEMORY_UTILIZATION': orjson.dumps(gpu_memory_utilization).decode(),
  })
  if quantize:
    os.environ['QUANTIZE'] = str(quantize)

  working_dir = os.path.abspath(os.path.dirname(__file__))

  try:
    server = subprocess.Popen(
      ['-m', 'bentoml', 'serve', '--working-dir', working_dir], executable=sys.executable, env=environ, text=True
    )
  finally:
    server.terminate()


if __name__ == '__main__':
  cli()
