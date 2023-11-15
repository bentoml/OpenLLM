# mypy: disable-error-code="misc"
from __future__ import annotations
import importlib.metadata
import inspect
import logging
import os
import string
import typing as t
from pathlib import Path

import fs
import fs.copy
import fs.errors
import orjson
from simple_di import Provide, inject

import bentoml
import openllm_core
from bentoml._internal.bento.build_config import BentoBuildConfig, DockerOptions, ModelSpec, PythonOptions
from bentoml._internal.configuration.containers import BentoMLContainer

from . import oci

if t.TYPE_CHECKING:
  from fs.base import FS

  import openllm
  from bentoml._internal.bento import BentoStore
  from bentoml._internal.models.model import ModelStore
  from openllm_core._typing_compat import (
    LiteralContainerRegistry,
    LiteralContainerVersionStrategy,
    LiteralSerialisation,
    LiteralString,
  )

logger = logging.getLogger(__name__)

OPENLLM_DEV_BUILD = 'OPENLLM_DEV_BUILD'


def build_editable(
  path: str, package: t.Literal['openllm', 'openllm_core', 'openllm_client'] = 'openllm'
) -> str | None:
  """Build OpenLLM if the OPENLLM_DEV_BUILD environment variable is set."""
  if not openllm_core.utils.check_bool_env(OPENLLM_DEV_BUILD, default=False):
    return None
  # We need to build the package in editable mode, so that we can import it
  from build import ProjectBuilder
  from build.env import IsolatedEnvBuilder

  module_location = openllm_core.utils.pkg.source_locations(package)
  if not module_location:
    raise RuntimeError(
      'Could not find the source location of OpenLLM. Make sure to unset OPENLLM_DEV_BUILD if you are developing OpenLLM.'
    )
  pyproject_path = Path(module_location).parent.parent / 'pyproject.toml'
  if os.path.isfile(pyproject_path.__fspath__()):
    logger.info('Generating built wheels for package %s...', package)
    with IsolatedEnvBuilder() as env:
      builder = ProjectBuilder(pyproject_path.parent)
      builder.python_executable = env.executable
      builder.scripts_dir = env.scripts_dir
      env.install(builder.build_system_requires)
      return builder.build('wheel', path, config_settings={'--global-option': '--quiet'})
  raise RuntimeError(
    'Custom OpenLLM build is currently not supported. Please install OpenLLM from PyPI or built it from Git source.'
  )


def construct_python_options(
  llm: openllm.LLM[t.Any, t.Any],
  llm_fs: FS,
  extra_dependencies: tuple[str, ...] | None = None,
  adapter_map: dict[str, str] | None = None,
) -> PythonOptions:
  packages = ['openllm', 'scipy']  # apparently bnb misses this one
  if adapter_map is not None:
    packages += ['openllm[fine-tune]']
  # NOTE: add openllm to the default dependencies
  # if users has openllm custom built wheels, it will still respect
  # that since bentoml will always install dependencies from requirements.txt
  # first, then proceed to install everything inside the wheels/ folder.
  if extra_dependencies is not None:
    packages += [f'openllm[{k}]' for k in extra_dependencies]

  req = llm.config['requirements']
  if req is not None:
    packages.extend(req)
  if str(os.environ.get('BENTOML_BUNDLE_LOCAL_BUILD', False)).lower() == 'false':
    packages.append(f"bentoml>={'.'.join([str(i) for i in openllm_core.utils.pkg.pkg_version_info('bentoml')])}")

  if not openllm_core.utils.is_torch_available():
    raise ValueError('PyTorch is not available. Make sure to have it locally installed.')
  packages.extend(
    ['torch==2.0.1+cu118', 'vllm==0.2.1.post1', 'xformers==0.0.22', 'bentoml[tracing]==1.1.9']
  )  # XXX: Currently locking this for correctness
  wheels: list[str] = []
  built_wheels = [
    build_editable(llm_fs.getsyspath('/'), t.cast(t.Literal['openllm', 'openllm_core', 'openllm_client'], p))
    for p in ('openllm_core', 'openllm_client', 'openllm')
  ]
  if all(i for i in built_wheels):
    wheels.extend([llm_fs.getsyspath(f"/{i.split('/')[-1]}") for i in t.cast(t.List[str], built_wheels)])
  return PythonOptions(
    packages=packages,
    wheels=wheels,
    lock_packages=False,
    extra_index_url=[
      'https://download.pytorch.org/whl/cu118',
      'https://huggingface.github.io/autogptq-index/whl/cu118/',
    ],
  )


def construct_docker_options(
  llm: openllm.LLM[t.Any, t.Any],
  _: FS,
  quantize: LiteralString | None,
  adapter_map: dict[str, str] | None,
  dockerfile_template: str | None,
  serialisation: LiteralSerialisation,
  container_registry: LiteralContainerRegistry,
  container_version_strategy: LiteralContainerVersionStrategy,
) -> DockerOptions:
  from openllm_cli._factory import parse_config_options

  environ = parse_config_options(llm.config, llm.config['timeout'], 1.0, None, True, os.environ.copy())
  env_dict = {
    'TORCH_DTYPE': str(llm._torch_dtype).split('.')[-1],
    'OPENLLM_BACKEND': llm.__llm_backend__,
    'OPENLLM_CONFIG': f"'{llm.config.model_dump_json(flatten=True).decode()}'",
    'OPENLLM_SERIALIZATION': serialisation,
    'BENTOML_DEBUG': str(True),
    'BENTOML_QUIET': str(False),
    'BENTOML_CONFIG_OPTIONS': f"'{environ['BENTOML_CONFIG_OPTIONS']}'",
    'TRUST_REMOTE_CODE': str(llm.trust_remote_code),
  }
  if adapter_map:
    env_dict['BITSANDBYTES_NOWELCOME'] = os.environ.get('BITSANDBYTES_NOWELCOME', '1')
  if llm._system_message:
    env_dict['OPENLLM_SYSTEM_MESSAGE'] = repr(llm._system_message)
  if llm._prompt_template:
    env_dict['OPENLLM_PROMPT_TEMPLATE'] = repr(llm._prompt_template.to_string())
  if quantize:
    env_dict['OPENLLM_QUANTIZE'] = str(quantize)
  return DockerOptions(
    base_image=f'{oci.get_base_container_name(container_registry)}:{oci.get_base_container_tag(container_version_strategy)}',
    env=env_dict,
    dockerfile_template=dockerfile_template,
  )


OPENLLM_MODEL_ID = '# openllm: model id'
OPENLLM_MODEL_TAG = '# openllm: model tag'
OPENLLM_MODEL_ADAPTER_MAP = '# openllm: model adapter map'


class _ServiceVarsFormatter(string.Formatter):
  keyword: LiteralString = '__model_name__'
  identifier: LiteralString = '# openllm: model name'

  def __init__(self, target: str):
    """The formatter that extends model_name to be formatted the 'service.py'."""
    super().__init__()
    self.target = target

  def vformat(self, format_string: str, *args: t.Any, **attrs: t.Any) -> t.Any:
    return super().vformat(format_string, (), {self.keyword: self.target})

  def can_format(self, value: str) -> bool:
    try:
      self.parse(value)
      return True
    except ValueError:
      return False

  def parse_line(self, line: str, nl: bool = True) -> str:
    if self.identifier not in line:
      return line
    gen = self.vformat(line)[: -(len(self.identifier) + 3)] + ('\n' if nl else '')
    return gen


class ModelIdFormatter(_ServiceVarsFormatter):
  keyword = '__model_id__'
  identifier = OPENLLM_MODEL_ID


class ModelTagFormatter(_ServiceVarsFormatter):
  keyword = '__model_tag__'
  identifier = OPENLLM_MODEL_TAG


class ModelAdapterMapFormatter(_ServiceVarsFormatter):
  keyword = '__model_adapter_map__'
  identifier = OPENLLM_MODEL_ADAPTER_MAP


_service_file = Path(os.path.abspath(__file__)).parent.parent / '_service.py'
_service_vars_file = Path(os.path.abspath(__file__)).parent.parent / '_service_vars_pkg.py'


def write_service(llm: openllm.LLM[t.Any, t.Any], adapter_map: dict[str, str] | None, llm_fs: FS) -> None:
  from openllm_core.utils import DEBUG

  model_id_formatter = ModelIdFormatter(llm.model_id)
  model_tag_formatter = ModelTagFormatter(str(llm.tag))
  adapter_map_formatter = ModelAdapterMapFormatter(orjson.dumps(adapter_map).decode())

  logger.debug(
    'Generating service vars file for %s at %s (dir=%s)', llm.model_id, '_service_vars.py', llm_fs.getsyspath('/')
  )
  with open(_service_vars_file.__fspath__(), 'r') as f:
    src_contents = f.readlines()
  for i, it in enumerate(src_contents):
    if model_id_formatter.identifier in it:
      src_contents[i] = model_id_formatter.parse_line(it)
    elif model_tag_formatter.identifier in it:
      src_contents[i] = model_tag_formatter.parse_line(it)
    elif adapter_map_formatter.identifier in it:
      src_contents[i] = adapter_map_formatter.parse_line(it)

  script = f"# GENERATED BY 'openllm build {llm.model_id}'. DO NOT EDIT\n\n" + ''.join(src_contents)
  if DEBUG:
    logger.info('Generated script:\n%s', script)
  llm_fs.writetext('_service_vars.py', script)

  logger.debug(
    'Generating service file for %s at %s (dir=%s)', llm.model_id, llm.config['service_name'], llm_fs.getsyspath('/')
  )
  with open(_service_file.__fspath__(), 'r') as f:
    service_src = f.read()
  llm_fs.writetext(llm.config['service_name'], service_src)


@inject
def create_bento(
  bento_tag: bentoml.Tag,
  llm_fs: FS,
  llm: openllm.LLM[t.Any, t.Any],
  quantize: LiteralString | None,
  dockerfile_template: str | None,
  adapter_map: dict[str, str] | None = None,
  extra_dependencies: tuple[str, ...] | None = None,
  serialisation: LiteralSerialisation | None = None,
  container_registry: LiteralContainerRegistry = 'ecr',
  container_version_strategy: LiteralContainerVersionStrategy = 'release',
  _bento_store: BentoStore = Provide[BentoMLContainer.bento_store],
  _model_store: ModelStore = Provide[BentoMLContainer.model_store],
) -> bentoml.Bento:
  _serialisation: LiteralSerialisation = openllm_core.utils.first_not_none(
    serialisation, default=llm.config['serialisation']
  )
  labels = dict(llm.identifying_params)
  labels.update(
    {
      '_type': llm.llm_type,
      '_framework': llm.__llm_backend__,
      'start_name': llm.config['start_name'],
      'base_name_or_path': llm.model_id,
      'bundler': 'openllm.bundle',
      **{
        f'{package.replace("-","_")}_version': importlib.metadata.version(package)
        for package in {'openllm', 'openllm-core', 'openllm-client'}
      },
    }
  )
  if adapter_map:
    labels.update(adapter_map)
  logger.debug("Building Bento '%s' with model backend '%s'", bento_tag, llm.__llm_backend__)
  # add service.py definition to this temporary folder
  write_service(llm, adapter_map, llm_fs)

  llm_spec = ModelSpec.from_item({'tag': str(llm.tag), 'alias': llm.tag.name})
  build_config = BentoBuildConfig(
    service=f"{llm.config['service_name']}:svc",
    name=bento_tag.name,
    labels=labels,
    models=[llm_spec],
    description=f"OpenLLM service for {llm.config['start_name']}",
    include=list(llm_fs.walk.files()),
    exclude=['/venv', '/.venv', '__pycache__/', '*.py[cod]', '*$py.class'],
    python=construct_python_options(llm, llm_fs, extra_dependencies, adapter_map),
    docker=construct_docker_options(
      llm,
      llm_fs,
      quantize,
      adapter_map,
      dockerfile_template,
      _serialisation,
      container_registry,
      container_version_strategy,
    ),
  )

  bento = bentoml.Bento.create(build_config=build_config, version=bento_tag.version, build_ctx=llm_fs.getsyspath('/'))
  # NOTE: the model_id_path here are only used for setting this environment variable within the container built with for BentoLLM.
  service_fs_path = fs.path.join('src', llm.config['service_name'])
  service_path = bento._fs.getsyspath(service_fs_path)
  with open(service_path, 'r') as f:
    service_contents = f.readlines()

  for it in service_contents:
    if '__bento_name__' in it:
      service_contents[service_contents.index(it)] = it.format(__bento_name__=str(bento.tag))

  script = ''.join(service_contents)
  if openllm_core.utils.DEBUG:
    logger.info('Generated script:\n%s', script)

  bento._fs.writetext(service_fs_path, script)
  if 'model_store' in inspect.signature(bento.save).parameters:
    return bento.save(bento_store=_bento_store, model_store=_model_store)
  # backward arguments. `model_store` is added recently
  return bento.save(bento_store=_bento_store)
