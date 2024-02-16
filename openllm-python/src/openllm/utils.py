import functools, importlib.metadata, openllm_core

__all__ = ['available_devices', 'device_count', 'generate_labels']


def generate_labels(serialisation):
  return {
    'framework': 'openllm',
    'serialisation': serialisation,
    **{package: importlib.metadata.version(package) for package in {'openllm', 'openllm-core', 'openllm-client'}},
  }


def available_devices():
  from ._strategies import NvidiaGpuResource

  return tuple(NvidiaGpuResource.from_system())


@functools.lru_cache(maxsize=1)
def device_count() -> int:
  return len(available_devices())


def __dir__():
  coreutils = set(dir(openllm_core.utils)) | set([it for it in openllm_core.utils._extras if not it.startswith('_')])
  return sorted(__all__) + sorted(list(coreutils))


def __getattr__(it):
  if hasattr(openllm_core.utils, it):
    return getattr(openllm_core.utils, it)
  raise AttributeError(f'module {__name__} has no attribute {it}')
