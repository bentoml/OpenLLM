import openllm_core


def __dir__():
  return dir(openllm_core.utils)


def __getattr__(name):
  if hasattr(openllm_core.utils, name):
    return getattr(openllm_core.utils, name)
  raise AttributeError(f'module {__name__} has no attribute {name}')
