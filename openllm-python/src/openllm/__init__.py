import logging as _logging, os as _os, pathlib as _pathlib, warnings as _warnings, typing as _t

from . import utils as utils

if utils.DEBUG:
  utils.set_debug_mode(True)
  _logging.basicConfig(level=_logging.NOTSET)
else:
  # configuration for bitsandbytes before import
  _os.environ['BITSANDBYTES_NOWELCOME'] = _os.environ.get('BITSANDBYTES_NOWELCOME', '1')
  # NOTE: The following warnings from bitsandbytes, and probably not that important for users to see when DEBUG is False
  _warnings.filterwarnings(
    'ignore', message='MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization'
  )
  _warnings.filterwarnings(
    'ignore', message='MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization'
  )
  _warnings.filterwarnings('ignore', message='The installed version of bitsandbytes was compiled without GPU support.')
  _warnings.filterwarnings(
    'ignore', message='Neither GITHUB_TOKEN nor GITHUB_JWT_TOKEN found: running as unauthenticated'
  )
COMPILED = _pathlib.Path(__file__).suffix in ('.pyd', '.so')
__lazy = utils.LazyModule(  # NOTE: update this to sys.modules[__name__] once mypy_extensions can recognize __spec__
  __name__,
  globals()['__file__'],
  {
    'exceptions': [],
    'client': ['HTTPClient', 'AsyncHTTPClient'],
    'bundle': [],
    'utils': ['api'],
    'serialisation': ['ggml', 'transformers', 'vllm'],
    '_llm': ['LLM'],
    '_deprecated': ['Runner'],
    '_runners': ['runner'],
    '_strategies': ['CascadingResourceStrategy', 'get_resource'],
  },
  extra_objects={'COMPILED': COMPILED},
)
__all__, __dir__ = __lazy.__all__, __lazy.__dir__

_BREAKING_INTERNAL = ['_service', '_service_vars']
_NEW_IMPL = ['LLM', *_BREAKING_INTERNAL]

if utils.pkg.pkg_version_info('bentoml') > (1, 2):
  import _openllm_tiny as _tiny
else:
  _tiny = None


def __getattr__(name: str) -> _t.Any:
  if name in _NEW_IMPL:
    if utils.getenv('IMPLEMENTATION', default='new_impl') == 'deprecated' or _tiny is None:
      if name in _BREAKING_INTERNAL:
        raise ImportError(
          f'"{name}" is an internal implementation and considered breaking with older OpenLLM. Please migrate your code if you depend on this.'
        )
      _warnings.warn(
        f'"{name}" is considered deprecated implementation and could be breaking. See https://github.com/bentoml/OpenLLM for more information on upgrading instruction.',
        DeprecationWarning,
        stacklevel=3,
      )
      return __lazy.__getattr__(name)
    else:
      return getattr(_tiny, name)
  else:
    return __lazy.__getattr__(name)
