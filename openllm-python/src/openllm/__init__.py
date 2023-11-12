"""OpenLLM.

An open platform for operating large language models in production. Fine-tune, serve,
deploy, and monitor any LLMs with ease.

* Built-in support for StableLM, Llama 2, Dolly, Flan-T5, Vicuna
* Option to bring your own fine-tuned LLMs
* Online Serving with HTTP, gRPC, SSE(coming soon) or custom API
* Native integration with BentoML and LangChain for custom LLM apps
"""

import logging as _logging
import os as _os
import pathlib as _pathlib
import warnings as _warnings

import openllm_cli as _cli

from openllm_cli import _sdk

from . import utils as utils


if utils.DEBUG:
  utils.set_debug_mode(True)
  utils.set_quiet_mode(False)
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
  # NOTE: ignore the following warning from ghapi as it is not important for users
  _warnings.filterwarnings(
    'ignore', message='Neither GITHUB_TOKEN nor GITHUB_JWT_TOKEN found: running as unauthenticated'
  )

COMPILED = _pathlib.Path(__file__).suffix in ('.pyd', '.so')

# NOTE: update this to sys.modules[__name__] once mypy_extensions can recognize __spec__
__lazy = utils.LazyModule(
  __name__,
  globals()['__file__'],
  {
    'exceptions': [],
    'client': ['HTTPClient', 'AsyncHTTPClient'],
    'bundle': [],
    'playground': [],
    'testing': [],
    'protocol': [],
    'utils': [],
    '_deprecated': ['Runner'],
    '_strategies': ['CascadingResourceStrategy', 'get_resource'],
    'entrypoints': ['mount_entrypoints'],
    'serialisation': ['ggml', 'transformers'],
    '_quantisation': ['infer_quantisation_config'],
    '_llm': ['LLM', 'LLMRunner', 'LLMRunnable'],
    '_generation': [
      'StopSequenceCriteria',
      'StopOnTokens',
      'LogitsProcessorList',
      'StoppingCriteriaList',
      'prepare_logits_processor',
    ],
  },
  extra_objects={
    'COMPILED': COMPILED,
    'cli': _cli,
    'start': _sdk.start,
    'start_grpc': _sdk.start_grpc,
    'build': _sdk.build,
    'import_model': _sdk.import_model,
    'list_models': _sdk.list_models,
  },
)
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__
