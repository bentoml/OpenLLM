import os
import typing as t

from openllm_core.utils import LazyModule

_import_structure = {
  '_package': ['create_bento', 'build_editable', 'construct_python_options', 'construct_docker_options'],
  'oci': [
    'CONTAINER_NAMES',
    'get_base_container_tag',
    'get_base_container_name',
    'supported_registries',
    'RefResolver',
  ],
}

if t.TYPE_CHECKING:
  from . import _package as _package, oci as oci
  from ._package import (
    build_editable as build_editable,
    construct_docker_options as construct_docker_options,
    construct_python_options as construct_python_options,
    create_bento as create_bento,
  )
  from .oci import (
    CONTAINER_NAMES as CONTAINER_NAMES,
    RefResolver as RefResolver,
    get_base_container_name as get_base_container_name,
    get_base_container_tag as get_base_container_tag,
    supported_registries as supported_registries,
  )

__lazy = LazyModule(__name__, os.path.abspath('__file__'), _import_structure)
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__
