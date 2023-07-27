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
import pathlib
import shutil
import subprocess
import typing as t
import uuid

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

_BUILDER = bentoml.container.get_backend("buildx")
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent

LiteralContainerRegistry = t.Literal["docker", "gh", "quay"]

_CONTAINER_REGISTRY: dict[LiteralContainerRegistry, str] = {"docker": "docker.io", "gh": "ghcr.io", "quay":"quay.io"}

_module_location = pkg.source_locations("openllm")

@functools.lru_cache
@apply(str.lower)
def get_base_container_name(local: bool) -> str:
    if local: return "openllm"
    try:
        repo = Repo(_module_location, search_parent_directories=True)
        url = repo.remotes.origin.url.rstrip(".git")
        is_http_url = url.startswith("https://")
        parts = url.split("/") if is_http_url else url.split(":")
        return f"{parts[-2]}/{parts[-1]}" if is_http_url else parts[-1]
    except InvalidGitRepositoryError: return f"openllm-{uuid.uuid4()}"

CONTAINER_NAMES = {alias: f"{reg}/{get_base_container_name(False)}" for alias, reg in _CONTAINER_REGISTRY.items()}

def get_base_container_tag() -> str:
    # To work with bare setup
    try: return Repo(_module_location, search_parent_directories=True).head.commit.hexsha
    # in this case, not a repo, then just generate GIT_SHA-like hash
    # from the root directory of openllm from this file
    except InvalidGitRepositoryError: return generate_hash_from_file(ROOT_DIR.resolve().__fspath__())

def build_container(registries: t.Literal["local"] | LiteralContainerRegistry | t.Sequence[LiteralContainerRegistry] | None = None, push: bool = False) -> None:
    try:
        if not _BUILDER.health(): raise Error
    except (Error, subprocess.CalledProcessError): raise RuntimeError("Building base container requires BuildKit (via Buildx) to be installed. See https://docs.docker.com/build/buildx/install/ for instalation instruction.") from None
    if device_count() == 0: raise RuntimeError("Building base container requires GPUs (None available)")
    if not shutil.which("nvidia-container-runtime"): raise RuntimeError("Make sure to have NVIDIA Container Toolkit setup correctly to compile CUDA kernel in container. See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html for more information.")
    if not _module_location: raise RuntimeError("Failed to determine source location of 'openllm'. (Possible broken installation)")
    pyproject_path = pathlib.Path(_module_location).parent.parent / "pyproject.toml"
    if not pyproject_path.exists(): raise ValueError("This utility can only be run within OpenLLM git repository. Clone it first with 'git clone https://github.com/bentoml/OpenLLM.git'")
    if registries == "local":
        tags = (f"{get_base_container_name(True)}:{get_base_container_tag()}", )
    elif registries is None: tags = tuple(f"{name}:{get_base_container_tag()}" for name in CONTAINER_NAMES)  # Default loop through all registry item
    else:
        if isinstance(registries, str): registries = [registries]
        else: registries = list(registries)
        tags = tuple(f"{CONTAINER_NAMES[name]}:{get_base_container_tag()}" for name in registries)
    try:
        _BUILDER.build(file=pathlib.Path(__file__).parent.joinpath("Dockerfile").resolve().__fspath__(), context_path=pyproject_path.parent.__fspath__(), tag=tags, push=push, progress="plain" if get_debug_mode() else "auto")
    except Exception as err: raise OpenLLMException("Failed to containerize base container images (Scroll up to see error above, or set OPENLLMDEVDEBUG=True for more traceback)") from err
