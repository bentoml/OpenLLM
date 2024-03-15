import importlib
from openllm_core.utils import LazyModule

_import_structure = {'openai': [], 'hf': [], 'cohere': []}


def mount_entrypoints(svc, llm):
  for module_name in _import_structure:
    module = importlib.import_module(f'.{module_name}', __name__)
    svc = module.mount_to_svc(svc, llm)
  return svc


__lazy = LazyModule(__name__, globals()['__file__'], _import_structure, extra_objects={'mount_entrypoints': mount_entrypoints})
__all__, __dir__, __getattr__ = __lazy.__all__, __lazy.__dir__, __lazy.__getattr__
