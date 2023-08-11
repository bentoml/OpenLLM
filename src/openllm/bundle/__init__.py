"""Build-related utilities. Some of these utilities are mainly used for 'openllm.build'.

These utilities will stay internal, and its API can be changed or updated without backward-compatibility.
"""
from __future__ import annotations
import sys
import typing as t

import openllm

_import_structure: dict[str, list[str]] = {"_package": ["create_bento", "build_editable", "construct_python_options", "construct_docker_options"], "oci": ["CONTAINER_NAMES", "get_base_container_tag", "build_container", "get_base_container_name", "supported_registries", "RefResolver"]}

if t.TYPE_CHECKING:
  from . import _package as _package
  from . import oci as oci
  from ._package import build_editable as build_editable
  from ._package import construct_docker_options as construct_docker_options
  from ._package import construct_python_options as construct_python_options
  from ._package import create_bento as create_bento
  from .oci import CONTAINER_NAMES as CONTAINER_NAMES
  from .oci import RefResolver as RefResolver
  from .oci import build_container as build_container
  from .oci import get_base_container_name as get_base_container_name
  from .oci import get_base_container_tag as get_base_container_tag
  from .oci import supported_registries as supported_registries
else:
  sys.modules[__name__] = openllm.utils.LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
