# mypy: disable-error-code="misc"
from __future__ import annotations
import importlib.metadata, inspect, logging, os, typing as t
from pathlib import Path
import fs, fs.copy, fs.errors, orjson, bentoml, openllm
from simple_di import Provide, inject
from bentoml._internal.bento.build_config import BentoBuildConfig, DockerOptions, ModelSpec, PythonOptions
from bentoml._internal.configuration.containers import BentoMLContainer
from . import oci

if t.TYPE_CHECKING:
  from fs.base import FS
  from openllm._typing_compat import LiteralString
  from bentoml._internal.bento import BentoStore
  from bentoml._internal.models.model import ModelStore
  from .oci import LiteralContainerRegistry, LiteralContainerVersionStrategy

logger = logging.getLogger(__name__)

OPENLLM_DEV_BUILD = "OPENLLM_DEV_BUILD"

def build_editable(path: str) -> str | None:
  """Build OpenLLM if the OPENLLM_DEV_BUILD environment variable is set."""
  if str(os.environ.get(OPENLLM_DEV_BUILD, False)).lower() != "true": return None
  # We need to build the package in editable mode, so that we can import it
  from build import ProjectBuilder
  from build.env import IsolatedEnvBuilder
  module_location = openllm.utils.pkg.source_locations("openllm")
  if not module_location: raise RuntimeError("Could not find the source location of OpenLLM. Make sure to unset OPENLLM_DEV_BUILD if you are developing OpenLLM.")
  pyproject_path = Path(module_location).parent.parent/"pyproject.toml"
  if os.path.isfile(pyproject_path.__fspath__()):
    logger.info("OpenLLM is installed in editable mode. Generating built wheels...")
    with IsolatedEnvBuilder() as env:
      builder = ProjectBuilder(pyproject_path.parent)
      builder.python_executable = env.executable
      builder.scripts_dir = env.scripts_dir
      env.install(builder.build_system_requires)
      return builder.build("wheel", path, config_settings={"--global-option": "--quiet"})
  raise RuntimeError("Custom OpenLLM build is currently not supported. Please install OpenLLM from PyPI or built it from Git source.")

def construct_python_options(llm: openllm.LLM[t.Any, t.Any], llm_fs: FS, extra_dependencies: tuple[str, ...] | None = None, adapter_map: dict[str, str | None] | None = None,) -> PythonOptions:
  packages = ["openllm", "scipy"]  # apparently bnb misses this one
  if adapter_map is not None: packages += ["openllm[fine-tune]"]
  # NOTE: add openllm to the default dependencies
  # if users has openllm custom built wheels, it will still respect
  # that since bentoml will always install dependencies from requirements.txt
  # first, then proceed to install everything inside the wheels/ folder.
  if extra_dependencies is not None: packages += [f"openllm[{k}]" for k in extra_dependencies]

  req = llm.config["requirements"]
  if req is not None: packages.extend(req)
  if str(os.environ.get("BENTOML_BUNDLE_LOCAL_BUILD", False)).lower() == "false": packages.append(f"bentoml>={'.'.join([str(i) for i in openllm.utils.pkg.pkg_version_info('bentoml')])}")

  env = llm.config["env"]
  framework_envvar = env["framework_value"]
  if framework_envvar == "flax":
    if not openllm.utils.is_flax_available(): raise ValueError(f"Flax is not available, while {env.framework} is set to 'flax'")
    packages.extend([importlib.metadata.version("flax"), importlib.metadata.version("jax"), importlib.metadata.version("jaxlib")])
  elif framework_envvar == "tf":
    if not openllm.utils.is_tf_available(): raise ValueError(f"TensorFlow is not available, while {env.framework} is set to 'tf'")
    candidates = ("tensorflow", "tensorflow-cpu", "tensorflow-gpu", "tf-nightly", "tf-nightly-cpu", "tf-nightly-gpu", "intel-tensorflow", "intel-tensorflow-avx512", "tensorflow-rocm", "tensorflow-macos",)
    # For the metadata, we have to look for both tensorflow and tensorflow-cpu
    for candidate in candidates:
      try:
        pkgver = importlib.metadata.version(candidate)
        if pkgver == candidate: packages.extend(["tensorflow"])
        else:
          _tf_version = importlib.metadata.version(candidate)
          packages.extend([f"tensorflow>={_tf_version}"])
        break
      except importlib.metadata.PackageNotFoundError: pass  # noqa: PERF203 # Ok to ignore here since we actually need to check for all possible tensorflow distribution.
  else:
    if not openllm.utils.is_torch_available(): raise ValueError("PyTorch is not available. Make sure to have it locally installed.")
    packages.extend([f'torch>={importlib.metadata.version("torch")}'])
  wheels: list[str] = []
  built_wheels = build_editable(llm_fs.getsyspath("/"))
  if built_wheels is not None: wheels.append(llm_fs.getsyspath(f"/{built_wheels.split('/')[-1]}"))
  return PythonOptions(packages=packages, wheels=wheels, lock_packages=False, extra_index_url=["https://download.pytorch.org/whl/cu118"])

def construct_docker_options(llm: openllm.LLM[t.Any, t.Any], _: FS, workers_per_resource: float, quantize: LiteralString | None, bettertransformer: bool | None, adapter_map: dict[str, str | None] | None, dockerfile_template: str | None, runtime: t.Literal["ggml", "transformers"], serialisation_format: t.Literal["safetensors", "legacy"], container_registry: LiteralContainerRegistry, container_version_strategy: LiteralContainerVersionStrategy) -> DockerOptions:
  from openllm.cli._factory import parse_config_options
  environ = parse_config_options(llm.config, llm.config["timeout"], workers_per_resource, None, True, os.environ.copy())
  env: openllm.utils.EnvVarMixin = llm.config["env"]
  if env["framework_value"] == "vllm": serialisation_format = "legacy"
  env_dict = {
      env.framework: env["framework_value"], env.config: f"'{llm.config.model_dump_json().decode()}'",
      env.model_id: f"/home/bentoml/bento/models/{llm.tag.path()}",
      "OPENLLM_MODEL": llm.config["model_name"], "OPENLLM_SERIALIZATION": serialisation_format,
      "OPENLLM_ADAPTER_MAP": f"'{orjson.dumps(adapter_map).decode()}'", "BENTOML_DEBUG": str(True), "BENTOML_QUIET": str(False), "BENTOML_CONFIG_OPTIONS": f"'{environ['BENTOML_CONFIG_OPTIONS']}'",
  }
  if adapter_map: env_dict["BITSANDBYTES_NOWELCOME"] = os.environ.get("BITSANDBYTES_NOWELCOME", "1")

  # We need to handle None separately here, as env from subprocess doesn't accept None value.
  _env = openllm.utils.EnvVarMixin(llm.config["model_name"], bettertransformer=bettertransformer, quantize=quantize, runtime=runtime)

  env_dict[_env.bettertransformer] = str(_env["bettertransformer_value"])
  if _env["quantize_value"] is not None: env_dict[_env.quantize] = t.cast(str, _env["quantize_value"])
  env_dict[_env.runtime] = _env["runtime_value"]
  return DockerOptions(base_image=f"{oci.CONTAINER_NAMES[container_registry]}:{oci.get_base_container_tag(container_version_strategy)}", env=env_dict, dockerfile_template=dockerfile_template)

@inject
def create_bento(bento_tag: bentoml.Tag, llm_fs: FS, llm: openllm.LLM[t.Any, t.Any], workers_per_resource: str | float, quantize: LiteralString | None, bettertransformer: bool | None, dockerfile_template: str | None, adapter_map: dict[str, str | None] | None = None, extra_dependencies: tuple[str, ...] | None = None,
                runtime: t.Literal[ "ggml", "transformers"] = "transformers", serialisation_format: t.Literal["safetensors", "legacy"] = "safetensors", container_registry: LiteralContainerRegistry = "ecr", container_version_strategy: LiteralContainerVersionStrategy = "release",
                _bento_store: BentoStore = Provide[BentoMLContainer.bento_store], _model_store: ModelStore = Provide[BentoMLContainer.model_store]) -> bentoml.Bento:
  framework_envvar = llm.config["env"]["framework_value"]
  labels = dict(llm.identifying_params)
  labels.update({"_type": llm.llm_type, "_framework": framework_envvar, "start_name": llm.config["start_name"], "base_name_or_path": llm.model_id, "bundler": "openllm.bundle"})
  if adapter_map: labels.update(adapter_map)
  if isinstance(workers_per_resource, str):
    if workers_per_resource == "round_robin": workers_per_resource = 1.0
    elif workers_per_resource == "conserved": workers_per_resource = 1.0 if openllm.utils.device_count() == 0 else float(1 / openllm.utils.device_count())
    else:
      try: workers_per_resource = float(workers_per_resource)
      except ValueError: raise ValueError("'workers_per_resource' only accept ['round_robin', 'conserved'] as possible strategies.") from None
  elif isinstance(workers_per_resource, int): workers_per_resource = float(workers_per_resource)
  logger.info("Building Bento for '%s'", llm.config["start_name"])
  # add service.py definition to this temporary folder
  openllm.utils.codegen.write_service(llm, adapter_map, llm_fs)

  llm_spec = ModelSpec.from_item({"tag": str(llm.tag), "alias": llm.tag.name})
  build_config = BentoBuildConfig(
      service=f"{llm.config['service_name']}:svc", name=bento_tag.name, labels=labels, description=f"OpenLLM service for {llm.config['start_name']}", include=list(llm_fs.walk.files()), exclude=["/venv", "/.venv", "__pycache__/", "*.py[cod]", "*$py.class"], python=construct_python_options(llm, llm_fs, extra_dependencies, adapter_map), models=[llm_spec],
      docker=construct_docker_options(llm, llm_fs, workers_per_resource, quantize, bettertransformer, adapter_map, dockerfile_template, runtime, serialisation_format, container_registry, container_version_strategy)
  )

  bento = bentoml.Bento.create(build_config=build_config, version=bento_tag.version, build_ctx=llm_fs.getsyspath("/"))
  # NOTE: the model_id_path here are only used for setting this environment variable within the container built with for BentoLLM.
  service_fs_path = fs.path.join("src", llm.config["service_name"])
  service_path = bento._fs.getsyspath(service_fs_path)
  with open(service_path, "r") as f:
    service_contents = f.readlines()

  for it in service_contents:
    if "__bento_name__" in it: service_contents[service_contents.index(it)] = it.format(__bento_name__=str(bento.tag))

  script = "".join(service_contents)
  if openllm.utils.DEBUG: logger.info("Generated script:\n%s", script)

  bento._fs.writetext(service_fs_path, script)
  if "model_store" in inspect.signature(bento.save).parameters: return bento.save(bento_store=_bento_store, model_store=_model_store)
  # backward arguments. `model_store` is added recently
  return bento.save(bento_store=_bento_store)
