from __future__ import annotations
import functools
import importlib
import logging
import os
import pathlib

import attr

from openllm_core._typing_compat import LiteralContainerVersionStrategy
from openllm_core.exceptions import OpenLLMException
from openllm_core.utils.lazy import VersionInfo

logger = logging.getLogger(__name__)

ROOT_DIR = pathlib.Path(os.path.abspath('__file__')).parent.parent.parent

_CONTAINER_REGISTRY = {
  'docker': 'docker.io/bentoml/openllm',
  'gh': 'ghcr.io/bentoml/openllm',
  'ecr': 'public.ecr.aws/y5w8i4y6/bentoml/openllm',
}

# TODO: support custom fork. Currently it only support openllm main.
_OWNER, _REPO = 'bentoml', 'openllm'


@attr.attrs(eq=False, order=False, slots=True, frozen=True)
class RefResolver:
  git_hash: str = attr.field()
  version: VersionInfo = attr.field(converter=lambda s: VersionInfo.from_version_string(s))
  strategy: LiteralContainerVersionStrategy = attr.field()

  @classmethod
  @functools.lru_cache(maxsize=64)
  def from_strategy(cls, strategy_or_version=None):
    # using default strategy
    if strategy_or_version is None or strategy_or_version == 'release':
      try:
        from ghapi.all import GhApi

        ghapi = GhApi(owner=_OWNER, repo=_REPO, authenticate=False)
        meta = ghapi.repos.get_latest_release()
        git_hash = ghapi.git.get_ref(ref=f"tags/{meta['name']}")['object']['sha']
      except Exception as err:
        raise OpenLLMException('Failed to determine latest release version.') from err
      return cls(git_hash=git_hash, version=meta['name'].lstrip('v'), strategy='release')
    elif strategy_or_version in ('latest', 'nightly'):  # latest is nightly
      return cls(git_hash='latest', version='0.0.0', strategy='latest')
    else:
      raise ValueError(f'Unknown strategy: {strategy_or_version}')

  @property
  def tag(self):
    return 'latest' if self.strategy in {'latest', 'nightly'} else repr(self.version)

  @staticmethod
  def construct_base_image(reg, strategy=None):
    if reg == 'gh':
      logger.warning("Setting base registry to 'gh' will affect cold start performance on GCP/AWS.")
    elif reg == 'docker':
      logger.warning('docker is base image is yet to be supported. Falling back to "ecr".')
      reg = 'ecr'
    return f'{_CONTAINER_REGISTRY[reg]}:{RefResolver.from_strategy(strategy).tag}'


__all__ = ['CONTAINER_NAMES', 'RefResolver', 'supported_registries']


def __dir__():
  return sorted(__all__)


def __getattr__(name):
  if name == 'supported_registries':
    return functools.lru_cache(1)(lambda: list(_CONTAINER_REGISTRY))()
  elif name == 'CONTAINER_NAMES':
    return _CONTAINER_REGISTRY
  elif name in __all__:
    return importlib.import_module('.' + name, __name__)
  else:
    raise AttributeError(f'{name} does not exists under {__name__}')
