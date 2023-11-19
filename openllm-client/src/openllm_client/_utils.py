import openllm_core


def __dir__():
  coreutils = set(dir(openllm_core.utils)) | set([it for it in openllm_core.utils._extras if not it.startswith('_')])
  return sorted(list(coreutils))


def __getattr__(name):
  if hasattr(openllm_core.utils, name):
    return getattr(openllm_core.utils, name)
  raise AttributeError(f'module {__name__} has no attribute {name}')
