from typing import Optional

import attr

from openllm_core._typing_compat import LiteralContainerRegistry, LiteralContainerVersionStrategy
from openllm_core.utils.lazy import VersionInfo

from . import _package as _package, oci as oci
from ._package import (
  build_editable as build_editable,
  construct_docker_options as construct_docker_options,
  construct_python_options as construct_python_options,
  create_bento as create_bento,
)

CONTAINER_NAMES: dict[LiteralContainerRegistry, str] = ...
supported_registries: list[str] = ...

@attr.attrs(eq=False, order=False, slots=True, frozen=True)
class RefResolver:
  git_hash: str
  version: VersionInfo
  strategy: LiteralContainerVersionStrategy

  @classmethod
  def from_strategy(cls, strategy_or_version: Optional[LiteralContainerVersionStrategy] = ...) -> RefResolver: ...
  @property
  def tag(self) -> str: ...
  @staticmethod
  def construct_base_image(
    reg: LiteralContainerRegistry, strategy: Optional[LiteralContainerVersionStrategy] = ...
  ) -> str: ...
