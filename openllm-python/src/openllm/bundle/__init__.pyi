from typing import Optional

import attr

from openllm_core._typing_compat import LiteralVersionStrategy
from openllm_core.utils.lazy import VersionInfo

from . import _package as _package
from ._package import (
  build_editable as build_editable,
  construct_docker_options as construct_docker_options,
  construct_python_options as construct_python_options,
  create_bento as create_bento,
)

@attr.attrs(eq=False, order=False, slots=True, frozen=True)
class RefResolver:
  git_hash: str
  version: VersionInfo
  strategy: LiteralVersionStrategy

  @classmethod
  def from_strategy(cls, strategy_or_version: Optional[LiteralVersionStrategy] = ...) -> RefResolver: ...
  @property
  def tag(self) -> str: ...
