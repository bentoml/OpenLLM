import logging as _logging, os as _os, pathlib as _pathlib, warnings as _warnings
from openllm_cli import _sdk
from . import utils as utils

if utils.DEBUG:
  utils.set_debug_mode(True)
  _logging.basicConfig(level=_logging.NOTSET)
else:
  # configuration for bitsandbytes before import
  _os.environ['BITSANDBYTES_NOWELCOME'] = _os.environ.get('BITSANDBYTES_NOWELCOME', '1')
  # NOTE: The following warnings from bitsandbytes, and probably not that important for users to see when DEBUG is False
  _warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization')
  _warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization')
  _warnings.filterwarnings('ignore', message='The installed version of bitsandbytes was compiled without GPU support.')
  _warnings.filterwarnings('ignore', message='Neither GITHUB_TOKEN nor GITHUB_JWT_TOKEN found: running as unauthenticated')
COMPILED = _pathlib.Path(__file__).suffix in ('.pyd', '.so')
__lazy = utils.LazyModule(  # NOTE: update this to sys.modules[__name__] once mypy_extensions can recognize __spec__
  __name__,
  globals()['__file__'],
  {
    'exceptions': [],
    'client': ['HTTPClient', 'AsyncHTTPClient'],
    'bundle': [],
    'testing': [],
    'protocol': [],
    'utils': [],
    '_deprecated': ['Runner'],
    '_strategies': ['CascadingResourceStrategy', 'get_resource'],
    'entrypoints': ['mount_entrypoints'],
    'serialisation': ['ggml', 'transformers'],
    '_quantisation': ['infer_quantisation_config'],
    '_llm': ['LLM'],
  },
  extra_objects={
    'COMPILED': COMPILED,
    'start': _sdk.start,
    'build': _sdk.build,  #
    'import_model': _sdk.import_model,
    'list_models': _sdk.list_models,  #
  },
)
__all__, __dir__, __getattr__ = __lazy.__all__, __lazy.__dir__, __lazy.__getattr__
