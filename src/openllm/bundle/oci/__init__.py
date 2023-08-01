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
import pathlib
import shutil
import subprocess
import typing as t

import git.cmd

import bentoml

from ...exceptions import Error
from ...exceptions import OpenLLMException
from ...utils import apply
from ...utils import device_count
from ...utils import get_debug_mode
from ...utils import pkg

_BUILDER = bentoml.container.get_backend("buildx")
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent

# TODO: support quay
LiteralContainerRegistry = t.Literal["docker", "gh", "ecr"]
LiteralContainerVersionStrategy = t.Literal["release", "nightly", "latest"]

# XXX: This registry will be hard code for now for easier to maintain
# but in the future, we can infer based on git repo and everything to make it more options for users
# to build the base image. For now, all of the base image will be <registry>/bentoml/openllm:...
# NOTE: The ECR registry is the public one and currently only @bentoml team has access to push it.
_CONTAINER_REGISTRY: dict[LiteralContainerRegistry, str] = {"docker": "docker.io/bentoml/openllm", "gh": "ghcr.io/bentoml/openllm", "ecr": "public.ecr.aws/y5w8i4y6/bentoml/openllm"}

_URI = "https://github.com/bentoml/openllm.git"

_module_location = pkg.source_locations("openllm")

@functools.lru_cache
@apply(str.lower)
def get_base_container_name(reg: LiteralContainerRegistry) -> str:
  return _CONTAINER_REGISTRY[reg]

@functools.lru_cache(maxsize=1)
def _git() -> git.cmd.Git:
  return git.cmd.Git(_URI)

@functools.lru_cache
def _nightly_ref() -> tuple[str, str]:
  return _git().ls_remote(_URI, "main", heads=True).split()

@functools.lru_cache
def _stable_ref() -> tuple[str, str]:
  return max([item.split() for item in _git().ls_remote(_URI, refs=True, tags=True).split("\n")], key=lambda tag: tuple(int(k) for k in tag[-1].replace("refs/tags/v", "").split(".")))

def get_base_container_tag(strategy: LiteralContainerVersionStrategy) -> str:
  if strategy == "release": return _stable_ref()[-1].replace("refs/tags/v", "")  # for stable, we can also use latest, but discouraged
  elif strategy == "latest": return "latest"
  elif strategy == "nightly": return f"sha-{_nightly_ref()[0][:7]}"  # we prefixed with sha-<git_rev_short> (giv_rev[:7])
  else: raise ValueError(f"Unknown strategy '{strategy}'. Valid strategies are 'release', 'nightly', and 'latest'")

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
__all__ = ["CONTAINER_NAMES", "get_base_container_tag", "build_container", "get_base_container_name", "supported_registries"]

def __dir__() -> list[str]:
  return sorted(__all__)

def __getattr__(name: str) -> t.Any:
  if name == "supported_registries": return functools.lru_cache(1)(lambda: list(_CONTAINER_REGISTRY))()
  elif name == "CONTAINER_NAMES": return _CONTAINER_REGISTRY
  elif name in __all__: return importlib.import_module("." + name, __name__)
  else: raise AttributeError(f"{name} does not exists under {__name__}")
