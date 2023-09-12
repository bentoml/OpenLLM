# mypy: disable-error-code="misc"
'''OCI-related utilities for OpenLLM. This module is considered to be internal and API are subjected to change.'''
from __future__ import annotations
import functools
import importlib
import logging
import os
import pathlib
import shutil
import subprocess
import typing as t

from datetime import datetime
from datetime import timedelta
from datetime import timezone

import attr
import orjson

import bentoml
import openllm
import openllm_core

from openllm_core.utils.lazy import VersionInfo

if t.TYPE_CHECKING:
  from ghapi import all

  from openllm_core._typing_compat import LiteralContainerRegistry
  from openllm_core._typing_compat import LiteralContainerVersionStrategy
  from openllm_core._typing_compat import LiteralString
  from openllm_core._typing_compat import RefTuple

all = openllm_core.utils.LazyLoader('all', globals(), 'ghapi.all')  # noqa: F811

logger = logging.getLogger(__name__)

_BUILDER = bentoml.container.get_backend('buildx')
ROOT_DIR = pathlib.Path(os.path.abspath('__file__')).parent.parent.parent

# XXX: This registry will be hard code for now for easier to maintain
# but in the future, we can infer based on git repo and everything to make it more options for users
# to build the base image. For now, all of the base image will be <registry>/bentoml/openllm:...
# NOTE: The ECR registry is the public one and currently only @bentoml team has access to push it.
_CONTAINER_REGISTRY: dict[LiteralContainerRegistry, str] = {'docker': 'docker.io/bentoml/openllm', 'gh': 'ghcr.io/bentoml/openllm', 'ecr': 'public.ecr.aws/y5w8i4y6/bentoml/openllm'}

# TODO: support custom fork. Currently it only support openllm main.
_OWNER = 'bentoml'
_REPO = 'openllm'

_module_location = openllm_core.utils.pkg.source_locations('openllm')

@functools.lru_cache
@openllm_core.utils.apply(str.lower)
def get_base_container_name(reg: LiteralContainerRegistry) -> str:
  return _CONTAINER_REGISTRY[reg]

def _convert_version_from_string(s: str) -> VersionInfo:
  return VersionInfo.from_version_string(s)

def _commit_time_range(r: int = 5) -> str:
  return (datetime.now(timezone.utc) - timedelta(days=r)).strftime('%Y-%m-%dT%H:%M:%SZ')

class VersionNotSupported(openllm.exceptions.OpenLLMException):
  """Raised when the stable release is too low that it doesn't include OpenLLM base container."""

_RefTuple: type[RefTuple] = openllm_core.utils.codegen.make_attr_tuple_class('_RefTuple', ['git_hash', 'version', 'strategy'])

def nightly_resolver(cls: type[RefResolver]) -> str:
  # NOTE: all openllm container will have sha-<git_hash[:7]>
  # This will use docker to run skopeo to determine the correct latest tag that is available
  # If docker is not found, then fallback to previous behaviour. Which the container might not exists.
  docker_bin = shutil.which('docker')
  if docker_bin is None:
    logger.warning(
        'To get the correct available nightly container, make sure to have docker available. Fallback to previous behaviour for determine nightly hash (container might not exists due to the lack of GPU machine at a time. See https://github.com/bentoml/OpenLLM/pkgs/container/openllm for available image.)'
    )
    commits = t.cast('list[dict[str, t.Any]]', cls._ghapi.repos.list_commits(since=_commit_time_range()))
    return next(f'sha-{it["sha"][:7]}' for it in commits if '[skip ci]' not in it['commit']['message'])
  # now is the correct behaviour
  return orjson.loads(subprocess.check_output([docker_bin, 'run', '--rm', '-it', 'quay.io/skopeo/stable:latest', 'list-tags', 'docker://ghcr.io/bentoml/openllm']).decode().strip())['Tags'][-2]

@attr.attrs(eq=False, order=False, slots=True, frozen=True)
class RefResolver:
  git_hash: str = attr.field()
  version: openllm_core.utils.VersionInfo = attr.field(converter=_convert_version_from_string)
  strategy: LiteralContainerVersionStrategy = attr.field()
  _ghapi: t.ClassVar[all.GhApi] = all.GhApi(owner=_OWNER, repo=_REPO, authenticate=False)

  @classmethod
  def _nightly_ref(cls) -> RefTuple:
    return _RefTuple((nightly_resolver(cls), 'refs/heads/main', 'nightly'))

  @classmethod
  def _release_ref(cls, version_str: str | None = None) -> RefTuple:
    _use_base_strategy = version_str is None
    if version_str is None:
      # NOTE: This strategy will only support openllm>0.2.12
      meta: dict[str, t.Any] = cls._ghapi.repos.get_latest_release()
      version_str = meta['name'].lstrip('v')
      version: tuple[str, str | None] = (cls._ghapi.git.get_ref(ref=f"tags/{meta['name']}")['object']['sha'], version_str)
    else:
      version = ('', version_str)
    if openllm_core.utils.VersionInfo.from_version_string(t.cast(str, version_str)) < (0, 2, 12):
      raise VersionNotSupported(f"Version {version_str} doesn't support OpenLLM base container. Consider using 'nightly' or upgrade 'openllm>=0.2.12'")
    return _RefTuple((*version, 'release' if _use_base_strategy else 'custom'))

  @classmethod
  @functools.lru_cache(maxsize=64)
  def from_strategy(cls, strategy_or_version: t.Literal['release', 'nightly'] | LiteralString | None = None) -> RefResolver:
    # using default strategy
    if strategy_or_version is None or strategy_or_version == 'release': return cls(*cls._release_ref())
    elif strategy_or_version == 'latest': return cls('latest', '0.0.0', 'latest')
    elif strategy_or_version == 'nightly':
      _ref = cls._nightly_ref()
      return cls(_ref[0], '0.0.0', _ref[-1])
    else:
      logger.warning('Using custom %s. Make sure that it is at lease 0.2.12 for base container support.', strategy_or_version)
      return cls(*cls._release_ref(version_str=strategy_or_version))

  @property
  def tag(self) -> str:
    # NOTE: latest tag can also be nightly, but discouraged to use it. For nightly refer to use sha-<git_hash_short>
    if self.strategy == 'latest': return 'latest'
    elif self.strategy == 'nightly': return self.git_hash
    else: return repr(self.version)

@functools.lru_cache(maxsize=256)
def get_base_container_tag(strategy: LiteralContainerVersionStrategy | None = None) -> str:
  return RefResolver.from_strategy(strategy).tag

def build_container(registries: LiteralContainerRegistry | t.Sequence[LiteralContainerRegistry] | None = None,
                    version_strategy: LiteralContainerVersionStrategy = 'release',
                    push: bool = False,
                    machine: bool = False) -> dict[str | LiteralContainerRegistry, str]:
  try:
    if not _BUILDER.health(): raise openllm.exceptions.Error
  except (openllm.exceptions.Error, subprocess.CalledProcessError):
    raise RuntimeError('Building base container requires BuildKit (via Buildx) to be installed. See https://docs.docker.com/build/buildx/install/ for instalation instruction.') from None
  if openllm_core.utils.device_count() == 0:
    raise RuntimeError('Building base container requires GPUs (None available)')
  if not shutil.which('nvidia-container-runtime'):
    raise RuntimeError('NVIDIA Container Toolkit is required to compile CUDA kernel in container.')
  if not _module_location:
    raise RuntimeError("Failed to determine source location of 'openllm'. (Possible broken installation)")
  pyproject_path = pathlib.Path(_module_location).parent.parent / 'pyproject.toml'
  if not pyproject_path.exists():
    raise ValueError("This utility can only be run within OpenLLM git repository. Clone it first with 'git clone https://github.com/bentoml/OpenLLM.git'")
  if not registries:
    tags: dict[str | LiteralContainerRegistry, str] = {
        alias: f'{value}:{get_base_container_tag(version_strategy)}' for alias, value in _CONTAINER_REGISTRY.items()
    }  # default to all registries with latest tag strategy
  else:
    registries = [registries] if isinstance(registries, str) else list(registries)
    tags = {name: f'{_CONTAINER_REGISTRY[name]}:{get_base_container_tag(version_strategy)}' for name in registries}
  try:
    outputs = _BUILDER.build(file=pathlib.Path(__file__).parent.joinpath('Dockerfile').resolve().__fspath__(),
                             context_path=pyproject_path.parent.__fspath__(),
                             tag=tuple(tags.values()),
                             push=push,
                             progress='plain' if openllm_core.utils.get_debug_mode() else 'auto',
                             quiet=machine)
    if machine and outputs is not None: tags['image_sha'] = outputs.decode('utf-8').strip()
  except Exception as err:
    raise openllm.exceptions.OpenLLMException(f'Failed to containerize base container images (Scroll up to see error above, or set OPENLLMDEVDEBUG=True for more traceback):\n{err}') from err
  return tags

if t.TYPE_CHECKING:
  CONTAINER_NAMES: dict[LiteralContainerRegistry, str]
  supported_registries: list[str]

__all__ = ['CONTAINER_NAMES', 'get_base_container_tag', 'build_container', 'get_base_container_name', 'supported_registries', 'RefResolver']

def __dir__() -> list[str]:
  return sorted(__all__)

def __getattr__(name: str) -> t.Any:
  if name == 'supported_registries': return functools.lru_cache(1)(lambda: list(_CONTAINER_REGISTRY))()
  elif name == 'CONTAINER_NAMES': return _CONTAINER_REGISTRY
  elif name in __all__: return importlib.import_module('.' + name, __name__)
  else: raise AttributeError(f'{name} does not exists under {__name__}')
