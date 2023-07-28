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

from git.exc import InvalidGitRepositoryError
from git.repo import Repo

import bentoml

from ...exceptions import Error
from ...exceptions import OpenLLMException
from ...utils import apply
from ...utils import device_count
from ...utils import generate_hash_from_file
from ...utils import get_debug_mode
from ...utils import pkg

if t.TYPE_CHECKING:
    from ..._types import P

_BUILDER = bentoml.container.get_backend("buildx")
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent

LiteralContainerRegistry = t.Literal["docker", "gh", "quay"]

_CONTAINER_REGISTRY: dict[LiteralContainerRegistry, str] = {"docker": "docker.io", "gh": "ghcr.io", "quay":"quay.io"}

_module_location = pkg.source_locations("openllm")

_R = t.TypeVar("_R")

def validate_correct_module_path(f: t.Callable[P, _R]) -> t.Callable[P, _R]:
    @functools.wraps(f)
    def inner(*args: P.args, **kwargs: P.kwargs) -> _R:
        if _module_location is None: raise RuntimeError("Failed to locate openllm installation. You either have a broken installation or something went wrong. Make sure to report back to the OpenLLM team.")
        return f(*args, **kwargs)
    return inner

@functools.lru_cache
@apply(str.lower)
@validate_correct_module_path
def get_base_container_name(local: bool) -> str:
    # this branch is already checked by decorator, hence it is here to make type checker happy
    def resolve_container_name() -> str:
        if t.TYPE_CHECKING: assert _module_location is not None
        return "pypi-openllm" if "site-packages" in _module_location else "openllm"
    if local: return resolve_container_name()
    try:
        repo = Repo(_module_location, search_parent_directories=True)
        url = repo.remotes.origin.url.rstrip(".git")
        is_http_url = url.startswith("https://")
        parts = url.split("/") if is_http_url else url.split(":")
        return f"{parts[-2]}/{parts[-1]}" if is_http_url else parts[-1]
    except InvalidGitRepositoryError: return resolve_container_name()

@functools.lru_cache
def get_registry_mapping() -> dict[LiteralContainerRegistry, str]: return {alias: f"{reg}/{get_base_container_name(False)}" for alias, reg in _CONTAINER_REGISTRY.items()}

@validate_correct_module_path
def get_base_container_tag() -> str:
    # To work with bare setup
    try: return Repo(_module_location, search_parent_directories=True).head.commit.hexsha
    # in this case, not a repo, then just generate GIT_SHA-like hash
    # from the root directory of openllm from this file
    except InvalidGitRepositoryError: return generate_hash_from_file(ROOT_DIR.resolve().__fspath__())

def build_container(registries: LiteralContainerRegistry | t.Sequence[LiteralContainerRegistry] | None = None, push: bool = False, machine: bool = False) -> dict[str | LiteralContainerRegistry, str]:
    try:
        if not _BUILDER.health(): raise Error
    except (Error, subprocess.CalledProcessError): raise RuntimeError("Building base container requires BuildKit (via Buildx) to be installed. See https://docs.docker.com/build/buildx/install/ for instalation instruction.") from None
    if device_count() == 0: raise RuntimeError("Building base container requires GPUs (None available)")
    if not shutil.which("nvidia-container-runtime"): raise RuntimeError("Make sure to have NVIDIA Container Toolkit setup correctly to compile CUDA kernel in container. See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html for more information.")
    if not _module_location: raise RuntimeError("Failed to determine source location of 'openllm'. (Possible broken installation)")
    pyproject_path = pathlib.Path(_module_location).parent.parent / "pyproject.toml"
    if not pyproject_path.exists(): raise ValueError("This utility can only be run within OpenLLM git repository. Clone it first with 'git clone https://github.com/bentoml/OpenLLM.git'")
    tags: dict[str | LiteralContainerRegistry, str]
    if not registries: tags = {alias: f"{name}:{get_base_container_tag()}" for alias, name in get_registry_mapping().items()}  # Default loop through all registry item
    else:
        if isinstance(registries, str): registries = [registries]
        else: registries = list(registries)
        tags = {name: f"{get_registry_mapping()[name]}:{get_base_container_tag()}" for name in registries}
    try:
        outputs = _BUILDER.build(file=pathlib.Path(__file__).parent.joinpath("Dockerfile").resolve().__fspath__(), context_path=pyproject_path.parent.__fspath__(), tag=tuple(tags.values()),
                                 push=push, progress="plain" if get_debug_mode() else "auto", quiet=machine)
        if machine and outputs is not None: tags["image_sha"] = outputs.decode("utf-8").strip()
    except Exception as err: raise OpenLLMException(f"Failed to containerize base container images (Scroll up to see error above, or set OPENLLMDEVDEBUG=True for more traceback):\n{err}") from err
    return tags

@functools.lru_cache
def _supported_registries() -> list[str]: return list(_CONTAINER_REGISTRY)

if t.TYPE_CHECKING:
    CONTAINER_NAMES: dict[LiteralContainerRegistry, str]
    supported_registries: list[str]

__all__ = ["CONTAINER_NAMES", "get_base_container_tag", "build_container", "get_base_container_name", "supported_registries"]
def __dir__() -> list[str]: return sorted(__all__)
def __getattr__(name: str) -> t.Any:
    if name == "supported_registries": return _supported_registries()
    elif name == "CONTAINER_NAMES": return get_registry_mapping()
    elif name in __all__: return importlib.import_module("." + name, __name__)
    else: raise AttributeError(f"{name} does not exists under {__name__}")
