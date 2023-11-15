# mypy: disable-error-code="misc"
from __future__ import annotations
import functools
import importlib
import logging
import os
import pathlib
import typing as t

import attr

from openllm_core.exceptions import OpenLLMException
from openllm_core.utils import codegen
from openllm_core.utils.lazy import VersionInfo

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import LiteralContainerRegistry, LiteralContainerVersionStrategy, RefTuple

logger = logging.getLogger(__name__)

ROOT_DIR = pathlib.Path(os.path.abspath('__file__')).parent.parent.parent

_CONTAINER_REGISTRY: dict[LiteralContainerRegistry, str] = {
  'docker': 'docker.io/bentoml/openllm',
  'gh': 'ghcr.io/bentoml/openllm',
  'ecr': 'public.ecr.aws/y5w8i4y6/bentoml/openllm',
}

# TODO: support custom fork. Currently it only support openllm main.
_OWNER, _REPO = 'bentoml', 'openllm'


def _convert_version_from_string(s: str) -> VersionInfo:
  return VersionInfo.from_version_string(s)


_RefTuple: type[RefTuple] = codegen.make_attr_tuple_class('_RefTuple', ['git_hash', 'version', 'strategy'])


@attr.attrs(eq=False, order=False, slots=True, frozen=True)
class RefResolver:
  git_hash: str = attr.field()
  version: VersionInfo = attr.field(converter=_convert_version_from_string)
  strategy: LiteralContainerVersionStrategy = attr.field()

  @classmethod
  def _release_ref(cls, version_str: str | None = None) -> RefTuple:
    try:
      from ghapi.all import GhApi

      ghapi = GhApi(owner=_OWNER, repo=_REPO, authenticate=False)
      meta = t.cast(t.Dict[str, t.Any], ghapi.repos.get_latest_release())
    except Exception as err:
      raise OpenLLMException('Failed to determine latest release version.') from err
    _use_base_strategy = version_str is None
    if version_str is None:
      # NOTE: This strategy will only support openllm>0.2.12
      version_str = meta['name'].lstrip('v')
      version = (ghapi.git.get_ref(ref=f"tags/{meta['name']}")['object']['sha'], version_str)
    else:
      version = ('', version_str)
    return _RefTuple((*version, 'release' if _use_base_strategy else 'custom'))

  @classmethod
  @functools.lru_cache(maxsize=64)
  def from_strategy(cls, strategy_or_version: LiteralContainerVersionStrategy | None = None) -> RefResolver:
    # using default strategy
    if strategy_or_version is None or strategy_or_version == 'release':
      return cls(*cls._release_ref())
    elif strategy_or_version in ('latest', 'nightly'):  # latest is nightly
      return cls(git_hash='latest', version='0.0.0', strategy='latest')
    else:
      raise ValueError(f'Unknown strategy: {strategy_or_version}')

  @property
  def tag(self) -> str:
    return 'latest' if self.strategy in {'latest', 'nightly'} else repr(self.version)


@functools.lru_cache(maxsize=256)
def get_base_container_tag(strategy: LiteralContainerVersionStrategy | None = None) -> str:
  return RefResolver.from_strategy(strategy).tag


def get_base_container_name(reg: LiteralContainerRegistry) -> str:
  return _CONTAINER_REGISTRY[reg]


if t.TYPE_CHECKING:
  CONTAINER_NAMES: dict[LiteralContainerRegistry, str]
  supported_registries: list[str]

__all__ = [
  'CONTAINER_NAMES',
  'get_base_container_tag',
  'get_base_container_name',
  'supported_registries',
  'RefResolver',
]


def __dir__() -> list[str]:
  return sorted(__all__)


def __getattr__(name: str) -> t.Any:
  if name == 'supported_registries':
    return functools.lru_cache(1)(lambda: list(_CONTAINER_REGISTRY))()
  elif name == 'CONTAINER_NAMES':
    return _CONTAINER_REGISTRY
  elif name in __all__:
    return importlib.import_module('.' + name, __name__)
  else:
    raise AttributeError(f'{name} does not exists under {__name__}')
