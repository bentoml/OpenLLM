# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OCI-related utilities for OpenLLM. This module is considered to be internal and API are subjected to change."""
from __future__ import annotations
import functools
import importlib
import logging
import pathlib
import shutil
import subprocess
import tempfile
import typing as t

import attr

import bentoml

from ...exceptions import Error
from ...exceptions import OpenLLMException
from ...utils import LazyLoader
from ...utils import VersionInfo
from ...utils import apply
from ...utils import device_count
from ...utils import get_debug_mode
from ...utils import pkg
from ...utils.codegen import make_attr_tuple_class

if t.TYPE_CHECKING:
  import git.cmd
  from git.repo.base import Repo

  from ..._types import RefTuple
else:
  git = LazyLoader("git", globals(), "git")
  git.cmd = LazyLoader("git.cmd", globals(), "git.cmd")
  Repo = LazyLoader("Repo", globals(), "git.repo.base.Repo")

logger = logging.getLogger(__name__)

_BUILDER = bentoml.container.get_backend("buildx")
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent

# TODO: support quay
LiteralContainerRegistry = t.Literal["docker", "gh", "ecr"]
LiteralContainerVersionStrategy = t.Literal["release", "nightly", "latest", "custom"]

# XXX: This registry will be hard code for now for easier to maintain
# but in the future, we can infer based on git repo and everything to make it more options for users
# to build the base image. For now, all of the base image will be <registry>/bentoml/openllm:...
# NOTE: The ECR registry is the public one and currently only @bentoml team has access to push it.
_CONTAINER_REGISTRY: dict[LiteralContainerRegistry, str] = {"docker": "docker.io/bentoml/openllm", "gh": "ghcr.io/bentoml/openllm", "ecr": "public.ecr.aws/y5w8i4y6/bentoml/openllm"}

# TODO: support custom fork. Currently it only support openllm main.
_URI = "https://github.com/bentoml/openllm.git"

_module_location = pkg.source_locations("openllm")

@functools.lru_cache
@apply(str.lower)
def get_base_container_name(reg: LiteralContainerRegistry) -> str:
  return _CONTAINER_REGISTRY[reg]

def _convert_version_from_string(s: str) -> VersionInfo:
  return VersionInfo.from_version_string(s)

class VersionNotSupported(OpenLLMException):
  """Raised when the stable release is too low that it doesn't include OpenLLM base container."""

_RefTuple: type[RefTuple] = make_attr_tuple_class("_RefTuple", ["git_hash", "version", "strategy"])

@attr.attrs(eq=False, order=False, slots=True, frozen=True)
class RefResolver:
  """TODO: Support offline mode.

  Maybe we need to save git hash when building the Bento.
  """
  git_hash: str = attr.field()
  version: VersionInfo = attr.field(converter=_convert_version_from_string)
  strategy: LiteralContainerVersionStrategy = attr.field()
  _git: git.cmd.Git = git.cmd.Git(_URI)  # TODO: support offline mode

  @classmethod
  @functools.lru_cache
  def nightly_resolver(cls) -> str:
    # Will do a clone bare to tempdir, and return the latest commit hash that we build the base image
    # NOTE: this is a bit expensive, but it is ok since we only run this during build
    with tempfile.TemporaryDirectory(prefix="openllm-bare-") as tempdir:
      cls._git.clone(_URI, tempdir, bare=True)
      return next(it.hexsha for it in Repo(tempdir).iter_commits("main", max_count=20) if "[skip ci]" not in str(it.summary))

  @classmethod
  def _nightly_ref(cls) -> RefTuple:
    return _RefTuple((cls.nightly_resolver(), "refs/heads/main", "nightly"))

  @classmethod
  def _release_ref(cls, version_str: str | None = None) -> RefTuple:
    _use_base_strategy = version_str is None
    if version_str is None:
      # NOTE: This strategy will only support openllm>0.2.12
      version = t.cast("tuple[str, str]", tuple(max((item.split() for item in cls._git.ls_remote(_URI, refs=True, tags=True).split("\n")), key=lambda tag: tuple(int(k) for k in t.cast("tuple[str, str]", tag)[-1].replace("refs/tags/v", "").split(".")))))
      version_str = version[-1].replace("refs/tags/v", "")
      version = (version[0], version_str)
    else:
      version = ("", version_str)
    if t.TYPE_CHECKING: assert version_str  # NOTE: Mypy cannot infer the correct type here. We have handle the cases where version_str is None in L86
    if VersionInfo.from_version_string(version_str) < (0, 2, 12): raise VersionNotSupported(f"Version {version_str} doesn't support OpenLLM base container. Consider using 'nightly' or upgrade 'openllm>=0.2.12'")
    return _RefTuple((*version, "release" if _use_base_strategy else "custom"))

  @classmethod
  def from_strategy(cls, strategy_or_version: t.Literal["release", "nightly"] | str | None = None) -> RefResolver:
    if strategy_or_version is None or strategy_or_version == "release":
      logger.debug("Using default strategy 'release' for resolving base image version.")
      return cls(*cls._release_ref())
    elif strategy_or_version == "latest":
      return cls("latest", "0.0.0", "latest")
    elif strategy_or_version == "nightly":
      _ref = cls._nightly_ref()
      return cls(_ref[0], "0.0.0", _ref[-1])
    else:
      logger.warning("Using custom %s. Make sure that it is at lease 0.2.12 for base container support.", strategy_or_version)
      return cls(*cls._release_ref(version_str=strategy_or_version))

  @property
  def tag(self) -> str:
    # NOTE: latest tag can also be nightly, but discouraged to use it. For nightly refer to use sha-<git_hash_short>
    if self.strategy == "latest": return "latest"
    elif self.strategy == "nightly": return f"sha-{self.git_hash[:7]}"
    else: return repr(self.version)

@functools.lru_cache(maxsize=256)
def get_base_container_tag(strategy: LiteralContainerVersionStrategy | None = None) -> str:
  return RefResolver.from_strategy(strategy).tag

def build_container(registries: LiteralContainerRegistry | t.Sequence[LiteralContainerRegistry] | None = None, version_strategy: LiteralContainerVersionStrategy = "release", push: bool = False, machine: bool = False) -> dict[str | LiteralContainerRegistry, str]:
  """This is a utility function for building base container for OpenLLM. It will build the base container for all registries if ``None`` is passed.

  Note that this is useful for debugging or for any users who wish to integrate vertically with OpenLLM. For most users, you should be able to get the image either from GitHub Container Registry or our public ECR registry.
  """
  try:
    if not _BUILDER.health(): raise Error
  except (Error, subprocess.CalledProcessError):
    raise RuntimeError("Building base container requires BuildKit (via Buildx) to be installed. See https://docs.docker.com/build/buildx/install/ for instalation instruction.") from None
  if device_count() == 0: raise RuntimeError("Building base container requires GPUs (None available)")
  if not shutil.which("nvidia-container-runtime"): raise RuntimeError("NVIDIA Container Toolkit is required to compile CUDA kernel in container.")
  if not _module_location: raise RuntimeError("Failed to determine source location of 'openllm'. (Possible broken installation)")
  pyproject_path = pathlib.Path(_module_location).parent.parent / "pyproject.toml"
  if not pyproject_path.exists(): raise ValueError("This utility can only be run within OpenLLM git repository. Clone it first with 'git clone https://github.com/bentoml/OpenLLM.git'")
  if t.TYPE_CHECKING: tags: dict[str | LiteralContainerRegistry, str]
  if not registries: tags = {alias: f"{value}:{get_base_container_tag(version_strategy)}" for alias, value in _CONTAINER_REGISTRY.items()}  # default to all registries with latest tag strategy
  else:
    registries = [registries] if isinstance(registries, str) else list(registries)
    tags = {name: f"{_CONTAINER_REGISTRY[name]}:{get_base_container_tag(version_strategy)}" for name in registries}
  try:
    outputs = _BUILDER.build(file=pathlib.Path(__file__).parent.joinpath("Dockerfile").resolve().__fspath__(), context_path=pyproject_path.parent.__fspath__(), tag=tuple(tags.values()), push=push, progress="plain" if get_debug_mode() else "auto", quiet=machine)
    if machine and outputs is not None: tags["image_sha"] = outputs.decode("utf-8").strip()
  except Exception as err:
    raise OpenLLMException(f"Failed to containerize base container images (Scroll up to see error above, or set OPENLLMDEVDEBUG=True for more traceback):\n{err}") from err
  return tags

if t.TYPE_CHECKING:
  CONTAINER_NAMES: dict[LiteralContainerRegistry, str]
  supported_registries: list[str]

__all__ = ["CONTAINER_NAMES", "get_base_container_tag", "build_container", "get_base_container_name", "supported_registries", "RefResolver"]

def __dir__() -> list[str]:
  return sorted(__all__)

def __getattr__(name: str) -> t.Any:
  if name == "supported_registries": return functools.lru_cache(1)(lambda: list(_CONTAINER_REGISTRY))()
  elif name == "CONTAINER_NAMES": return _CONTAINER_REGISTRY
  elif name in __all__: return importlib.import_module("." + name, __name__)
  else: raise AttributeError(f"{name} does not exists under {__name__}")
