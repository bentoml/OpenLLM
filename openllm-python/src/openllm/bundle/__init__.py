import os

from openllm_core.utils import LazyModule

__lazy = LazyModule(
  __name__,
  os.path.abspath('__file__'),
  {
    '_package': ['create_bento', 'build_editable', 'construct_python_options', 'construct_docker_options'],
    'oci': ['CONTAINER_NAMES', 'supported_registries', 'RefResolver'],
  },
)
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__
