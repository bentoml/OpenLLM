from __future__ import annotations
import itertools, logging, os, re, subprocess, sys, typing as t
from simple_di import Provide, inject
import bentoml, openllm_core, orjson
from bentoml._internal.configuration.containers import BentoMLContainer
from openllm_core._typing_compat import LiteralSerialisation
from openllm_core.exceptions import OpenLLMException
from openllm_core.utils import WARNING_ENV_VAR, codegen, first_not_none, get_disable_warnings, is_vllm_available

if t.TYPE_CHECKING:
  from bentoml._internal.bento import BentoStore
  from openllm_core._configuration import LLMConfig
  from openllm_core._typing_compat import LiteralBackend, LiteralQuantise, LiteralString

logger = logging.getLogger(__name__)


def _start(
  model_id: str,
  timeout: int = 30,
  workers_per_resource: t.Literal['conserved', 'round_robin'] | float | None = None,
  device: tuple[str, ...] | t.Literal['all'] | None = None,
  quantize: LiteralQuantise | None = None,
  adapter_map: dict[LiteralString, str | None] | None = None,
  backend: LiteralBackend | None = None,
  additional_args: list[str] | None = None,
  cors: bool = False,
  __test__: bool = False,
  **_: t.Any,
) -> LLMConfig | subprocess.Popen[bytes]:
  """Python API to start a LLM server. These provides one-to-one mapping to CLI arguments.

  For all additional arguments, pass it as string to ``additional_args``. For example, if you want to
  pass ``--port 5001``, you can pass ``additional_args=["--port", "5001"]``

  > [!NOTE] This will create a blocking process, so if you use this API, you can create a running sub thread
  > to start the server instead of blocking the main thread.

  ``openllm.start`` will invoke ``click.Command`` under the hood, so it behaves exactly the same as the CLI interaction.

  Args:
    model_id: The model id to start this LLMServer
    timeout: The server timeout
    workers_per_resource: Number of workers per resource assigned.
                          See [resource scheduling](https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy)
                          for more information. By default, this is set to 1.

                          > [!NOTE] ``--workers-per-resource`` will also accept the following strategies:
                          > - ``round_robin``: Similar behaviour when setting ``--workers-per-resource 1``. This is useful for smaller models.
                          > - ``conserved``: This will determine the number of available GPU resources, and only assign
                          >                  one worker for the LLMRunner. For example, if ther are 4 GPUs available, then ``conserved`` is
                          >                  equivalent to ``--workers-per-resource 0.25``.
    device: Assign GPU devices (if available) to this LLM. By default, this is set to ``None``. It also accepts 'all'
    argument to assign all available GPUs to this LLM.
    quantize: Quantize the model weights. This is only applicable for PyTorch models.
              Possible quantisation strategies:
              - int8: Quantize the model with 8bit (bitsandbytes required)
              - int4: Quantize the model with 4bit (bitsandbytes required)
              - gptq: Quantize the model with GPTQ (auto-gptq required)
    cors: Whether to enable CORS for this LLM. By default, this is set to ``False``.
    adapter_map: The adapter mapping of LoRA to use for this LLM. It accepts a dictionary of ``{adapter_id: adapter_name}``.
    backend: The backend to use for this LLM. By default, this is set to ``pt``.
    additional_args: Additional arguments to pass to ``openllm start``.
  """
  from .entrypoint import start_command

  os.environ['BACKEND'] = openllm_core.utils.first_not_none(backend, default='vllm' if is_vllm_available() else 'pt')
  args: list[str] = [model_id]
  if timeout:
    args.extend(['--server-timeout', str(timeout)])
  if workers_per_resource:
    args.extend(['--workers-per-resource', str(workers_per_resource) if not isinstance(workers_per_resource, str) else workers_per_resource])
  if device and not os.environ.get('CUDA_VISIBLE_DEVICES'):
    args.extend(['--device', ','.join(device)])
  if quantize:
    args.extend(['--quantize', str(quantize)])
  if cors:
    args.append('--cors')
  if adapter_map:
    args.extend(list(itertools.chain.from_iterable([['--adapter-id', f"{k}{':'+v if v else ''}"] for k, v in adapter_map.items()])))
  if additional_args:
    args.extend(additional_args)
  if __test__:
    args.append('--return-process')

  cmd = start_command
  return cmd.main(args=args, standalone_mode=False)


@inject
def _build(
  model_id: str,
  model_version: str | None = None,
  bento_version: str | None = None,
  quantize: LiteralQuantise | None = None,
  adapter_map: dict[str, str | None] | None = None,
  build_ctx: str | None = None,
  enable_features: tuple[str, ...] | None = None,
  dockerfile_template: str | None = None,
  overwrite: bool = False,
  push: bool = False,
  force_push: bool = False,
  containerize: bool = False,
  serialisation: LiteralSerialisation | None = None,
  additional_args: list[str] | None = None,
  bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
) -> bentoml.Bento:
  """Package a LLM into a BentoLLM.

  The LLM will be built into a BentoService with the following structure:
  if ``quantize`` is passed, it will instruct the model to be quantized dynamically during serving time.

  ``openllm.build`` will invoke ``click.Command`` under the hood, so it behaves exactly the same as ``openllm build`` CLI.

  Args:
    model_id: The model id to build this BentoLLM
    model_version: Optional model version for this given LLM
    bento_version: Optional bento veresion for this given BentoLLM
    quantize: Quantize the model weights. This is only applicable for PyTorch models.
              Possible quantisation strategies:
              - int8: Quantize the model with 8bit (bitsandbytes required)
              - int4: Quantize the model with 4bit (bitsandbytes required)
              - gptq: Quantize the model with GPTQ (auto-gptq required)
    adapter_map: The adapter mapping of LoRA to use for this LLM. It accepts a dictionary of ``{adapter_id: adapter_name}``.
    build_ctx: The build context to use for building BentoLLM. By default, it sets to current directory.
    enable_features: Additional OpenLLM features to be included with this BentoLLM.
    dockerfile_template: The dockerfile template to use for building BentoLLM. See https://docs.bentoml.com/en/latest/guides/containerization.html#dockerfile-template.
    overwrite: Whether to overwrite the existing BentoLLM. By default, this is set to ``False``.
    push: Whether to push the result bento to BentoCloud. Make sure to login with 'bentoml cloud login' first.
    containerize: Whether to containerize the Bento after building. '--containerize' is the shortcut of 'openllm build && bentoml containerize'.
                  Note that 'containerize' and 'push' are mutually exclusive
                  container_registry: Container registry to choose the base OpenLLM container image to build from. Default to ECR.
    serialisation: Serialisation for saving models. Default to 'safetensors', which is equivalent to `safe_serialization=True`
    additional_args: Additional arguments to pass to ``openllm build``.
    bento_store: Optional BentoStore for saving this BentoLLM. Default to the default BentoML local store.

  Returns:
      ``bentoml.Bento | str``: BentoLLM instance. This can be used to serve the LLM or can be pushed to BentoCloud.
  """
  from openllm.serialisation.transformers.weights import has_safetensors_weights

  args: list[str] = [
    sys.executable,
    '-m',
    'openllm',
    'build',
    model_id,
    '--machine',
    '--quiet',
    '--serialisation',
    first_not_none(serialisation, default='safetensors' if has_safetensors_weights(model_id, model_version) else 'legacy'),
  ]
  if quantize:
    args.extend(['--quantize', quantize])
  if containerize and push:
    raise OpenLLMException("'containerize' and 'push' are currently mutually exclusive.")
  if push:
    args.extend(['--push'])
  if containerize:
    args.extend(['--containerize'])
  if build_ctx:
    args.extend(['--build-ctx', build_ctx])
  if enable_features:
    args.extend([f'--enable-features={f}' for f in enable_features])
  if overwrite:
    args.append('--overwrite')
  if adapter_map:
    args.extend([f"--adapter-id={k}{':'+v if v is not None else ''}" for k, v in adapter_map.items()])
  if model_version:
    args.extend(['--model-version', model_version])
  if bento_version:
    args.extend(['--bento-version', bento_version])
  if dockerfile_template:
    args.extend(['--dockerfile-template', dockerfile_template])
  if additional_args:
    args.extend(additional_args)
  if force_push:
    args.append('--force-push')

  current_disable_warning = get_disable_warnings()
  os.environ[WARNING_ENV_VAR] = str(True)
  try:
    output = subprocess.check_output(args, env=os.environ.copy(), cwd=build_ctx or os.getcwd())
  except subprocess.CalledProcessError as e:
    logger.error("Exception caught while building Bento for '%s'", model_id, exc_info=e)
    if e.stderr:
      raise OpenLLMException(e.stderr.decode('utf-8')) from None
    raise OpenLLMException(str(e)) from None
  matched = re.match(r'__object__:(\{.*\})$', output.decode('utf-8').strip())
  if matched is None:
    raise ValueError(
      f"Failed to find tag from output: {output.decode('utf-8').strip()}\nNote: Output from 'openllm build' might not be correct. Please open an issue on GitHub."
    )
  os.environ[WARNING_ENV_VAR] = str(current_disable_warning)
  try:
    result = orjson.loads(matched.group(1))
  except orjson.JSONDecodeError as e:
    raise ValueError(
      f"Failed to decode JSON from output: {output.decode('utf-8').strip()}\nNote: Output from 'openllm build' might not be correct. Please open an issue on GitHub."
    ) from e
  return bentoml.get(result['tag'], _bento_store=bento_store)


def _import_model(
  model_id: str,
  model_version: str | None = None,
  backend: LiteralBackend | None = None,
  quantize: LiteralQuantise | None = None,
  serialisation: LiteralSerialisation | None = None,
  additional_args: t.Sequence[str] | None = None,
) -> dict[str, t.Any]:
  """Import a LLM into local store.

  > [!NOTE]
  > If ``quantize`` is passed, the model weights will be saved as quantized weights. You should
  > only use this option if you want the weight to be quantized by default. Note that OpenLLM also
  > support on-demand quantisation during initial startup.

  ``openllm.import_model`` will invoke ``click.Command`` under the hood, so it behaves exactly the same as the CLI ``openllm import``.

  > [!NOTE]
  > ``openllm.start`` will automatically invoke ``openllm.import_model`` under the hood.

  Args:
    model_id: required model id for this given LLM
    model_version: Optional model version for this given LLM
    backend: The backend to use for this LLM. By default, this is set to ``pt``.
    quantize: Quantize the model weights. This is only applicable for PyTorch models.
              Possible quantisation strategies:
              - int8: Quantize the model with 8bit (bitsandbytes required)
              - int4: Quantize the model with 4bit (bitsandbytes required)
              - gptq: Quantize the model with GPTQ (auto-gptq required)
    serialisation: Type of model format to save to local store. If set to 'safetensors', then OpenLLM will save model using safetensors. Default behaviour is similar to ``safe_serialization=False``.
    additional_args: Additional arguments to pass to ``openllm import``.

  Returns:
    ``bentoml.Model``:BentoModel of the given LLM. This can be used to serve the LLM or can be pushed to BentoCloud.
  """
  from .entrypoint import import_command

  args = [model_id, '--quiet']
  if backend is not None:
    args.extend(['--backend', backend])
  if model_version is not None:
    args.extend(['--model-version', str(model_version)])
  if quantize is not None:
    args.extend(['--quantize', quantize])
  if serialisation is not None:
    args.extend(['--serialisation', serialisation])
  if additional_args is not None:
    args.extend(additional_args)
  return import_command.main(args=args, standalone_mode=False)


def _list_models() -> dict[str, t.Any]:
  """List all available models within the local store."""
  from .entrypoint import models_command

  return models_command.main(args=['--quiet'], standalone_mode=False)


start, build, import_model, list_models = (
  codegen.gen_sdk(_start),
  codegen.gen_sdk(_build),
  codegen.gen_sdk(_import_model),
  codegen.gen_sdk(_list_models),
)
__all__ = ['start', 'build', 'import_model', 'list_models']
