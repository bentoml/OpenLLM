"""Build-related utilities. Some of these utilities are mainly used for 'openllm.build'.

These utilities will stay internal, and its API can be changed or updated without backward-compatibility.
"""
from __future__ import annotations
import sys
import typing as t

import openllm

_import_structure: dict[str, list[str]] = {"_package": ["create_bento", "build_editable", "construct_python_options", "construct_docker_options"], "oci": ["CONTAINER_NAMES", "get_base_container_tag", "build_container", "get_base_container_name", "supported_registries", "RefResolver"]}

if t.TYPE_CHECKING:
  from . import (
    _package as _package,
    oci as oci,
  )
  from ._package import (
    build_editable as build_editable,
    construct_docker_options as construct_docker_options,
    construct_python_options as construct_python_options,
    create_bento as create_bento,
  )
  from .oci import (
    CONTAINER_NAMES as CONTAINER_NAMES,
    RefResolver as RefResolver,
    build_container as build_container,
    get_base_container_name as get_base_container_name,
    get_base_container_tag as get_base_container_tag,
    supported_registries as supported_registries,
  )
else:
  sys.modules[__name__] = openllm.utils.LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
