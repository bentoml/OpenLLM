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

"""Build-related utilities. Some of these utilities are mainly used for 'openllm.build'.

These utilities will stay internal, and its API can be changed or updated without backward-compatibility.
"""
from __future__ import annotations
import sys
import typing as t

from . import oci as oci
from ..utils import LazyModule

_import_structure: dict[str, list[str]] = {
    "_package": ["create_bento", "build_editable", "construct_python_options", "construct_docker_options"],
    "oci": oci.__all__,
}

if t.TYPE_CHECKING:
    from . import _package as _package
    from ._package import build_editable as build_editable
    from ._package import construct_docker_options as construct_docker_options
    from ._package import construct_python_options as construct_python_options
    from ._package import create_bento as create_bento
    from .oci import CONTAINER_NAMES as CONTAINER_NAMES
    from .oci import build_container as build_container
    from .oci import get_base_container_name as get_base_container_name
    from .oci import get_base_container_tag as get_base_container_tag
    from .oci import supported_registries as supported_registries
else: sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
