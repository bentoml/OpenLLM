from __future__ import annotations
import os, attr, functools
from openllm_core._typing_compat import LiteralVersionStrategy
from openllm_core.exceptions import OpenLLMException
from openllm_core.utils.lazy import VersionInfo, LazyModule


@attr.attrs(eq=False, order=False, slots=True, frozen=True)
class RefResolver:
  git_hash: str = attr.field()
  version: VersionInfo = attr.field(converter=lambda s: VersionInfo.from_version_string(s))
  strategy: LiteralVersionStrategy = attr.field()

  @classmethod
  @functools.lru_cache(maxsize=64)
  def from_strategy(cls, strategy_or_version: LiteralVersionStrategy | None = None) -> RefResolver:
    # using default strategy
    if strategy_or_version is None or strategy_or_version == 'release':
      try:
        from ghapi.all import GhApi

        ghapi = GhApi(owner='bentoml', repo='openllm', authenticate=False)
        meta = ghapi.repos.get_latest_release()
        git_hash = ghapi.git.get_ref(ref=f"tags/{meta['name']}")['object']['sha']
      except Exception as err:
        raise OpenLLMException('Failed to determine latest release version.') from err
      return cls(git_hash, meta['name'].lstrip('v'), 'release')
    elif strategy_or_version in ('latest', 'nightly'):  # latest is nightly
      return cls('latest', '0.0.0', 'latest')
    else:
      raise ValueError(f'Unknown strategy: {strategy_or_version}')

  @property
  def tag(self) -> str:
    return 'latest' if self.strategy in {'latest', 'nightly'} else repr(self.version)


__lazy = LazyModule(
  __name__,
  os.path.abspath('__file__'),  #
  {'_package': ['create_bento', 'build_editable', 'construct_python_options', 'construct_docker_options']},
  extra_objects={'RefResolver': RefResolver},
)
__all__, __dir__, __getattr__ = __lazy.__all__, __lazy.__dir__, __lazy.__getattr__
