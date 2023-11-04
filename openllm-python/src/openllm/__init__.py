'''OpenLLM.

An open platform for operating large language models in production. Fine-tune, serve,
deploy, and monitor any LLMs with ease.

* Built-in support for StableLM, Llama 2, Dolly, Flan-T5, Vicuna
* Option to bring your own fine-tuned LLMs
* Online Serving with HTTP, gRPC, SSE(coming soon) or custom API
* Native integration with BentoML and LangChain for custom LLM apps
'''
from __future__ import annotations
import logging as _logging
import os as _os
import typing as _t
import warnings as _warnings

from pathlib import Path as _Path

import openllm_core

from openllm_core._configuration import GenerationConfig as GenerationConfig
from openllm_core._configuration import LLMConfig as LLMConfig
from openllm_core._configuration import SamplingParams as SamplingParams
from openllm_core._schema import GenerateInput as GenerateInput
from openllm_core._schema import GenerationOutput as GenerationOutput
from openllm_core._schema import HfAgentInput as HfAgentInput
from openllm_core._schema import MetadataOutput as MetadataOutput
from openllm_core.config import CONFIG_MAPPING as CONFIG_MAPPING
from openllm_core.config import CONFIG_MAPPING_NAMES as CONFIG_MAPPING_NAMES
from openllm_core.config import AutoConfig as AutoConfig
from openllm_core.config import BaichuanConfig as BaichuanConfig
from openllm_core.config import ChatGLMConfig as ChatGLMConfig
from openllm_core.config import DollyV2Config as DollyV2Config
from openllm_core.config import FalconConfig as FalconConfig
from openllm_core.config import FlanT5Config as FlanT5Config
from openllm_core.config import GPTNeoXConfig as GPTNeoXConfig
from openllm_core.config import LlamaConfig as LlamaConfig
from openllm_core.config import MPTConfig as MPTConfig
from openllm_core.config import OPTConfig as OPTConfig
from openllm_core.config import StableLMConfig as StableLMConfig
from openllm_core.config import StarCoderConfig as StarCoderConfig

from . import exceptions as exceptions
from . import utils as utils

if openllm_core.utils.DEBUG:
  openllm_core.utils.set_debug_mode(True)
  openllm_core.utils.set_quiet_mode(False)
  _logging.basicConfig(level=_logging.NOTSET)
else:
  # configuration for bitsandbytes before import
  _os.environ['BITSANDBYTES_NOWELCOME'] = _os.environ.get('BITSANDBYTES_NOWELCOME', '1')
  # NOTE: The following warnings from bitsandbytes, and probably not that important for users to see when DEBUG is False
  _warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization')
  _warnings.filterwarnings('ignore', message='MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization')
  _warnings.filterwarnings('ignore', message='The installed version of bitsandbytes was compiled without GPU support.')
  # NOTE: ignore the following warning from ghapi as it is not important for users
  _warnings.filterwarnings('ignore', message='Neither GITHUB_TOKEN nor GITHUB_JWT_TOKEN found: running as unauthenticated')

_import_structure: dict[str, list[str]] = {
    'exceptions': [],
    'client': [],
    'bundle': [],
    'playground': [],
    'testing': [],
    'prompts': ['PromptTemplate'],
    'protocol': [],
    'entrypoints': [],
    'utils': [],
    'serialisation': ['ggml', 'transformers'],
    'cli._sdk': ['start', 'start_grpc', 'build', 'import_model', 'list_models'],
    '_quantisation': ['infer_quantisation_config'],
    '_llm': ['LLM', 'Runner', 'LLMRunner', 'LLMRunnable'],
    '_generation': ['StopSequenceCriteria', 'StopOnTokens', 'LogitsProcessorList', 'StoppingCriteriaList', 'prepare_logits_processor'],
}
COMPILED = _Path(__file__).suffix in ('.pyd', '.so')

if _t.TYPE_CHECKING:
  from . import bundle as bundle
  from . import cli as cli
  from . import client as client
  from . import playground as playground
  from . import serialisation as serialisation
  from . import testing as testing
  from ._generation import LogitsProcessorList as LogitsProcessorList
  from ._generation import StopOnTokens as StopOnTokens
  from ._generation import StoppingCriteriaList as StoppingCriteriaList
  from ._generation import StopSequenceCriteria as StopSequenceCriteria
  from ._generation import prepare_logits_processor as prepare_logits_processor
  from ._llm import LLM as LLM
  from ._llm import LLMRunnable as LLMRunnable
  from ._llm import LLMRunner as LLMRunner
  from ._llm import Runner as Runner
  from ._quantisation import infer_quantisation_config as infer_quantisation_config
  from .cli._sdk import build as build
  from .cli._sdk import import_model as import_model
  from .cli._sdk import list_models as list_models
  from .cli._sdk import start as start
  from .cli._sdk import start_grpc as start_grpc
  from .prompts import PromptTemplate as PromptTemplate
  from .protocol import openai as openai
  from .serialisation import ggml as ggml
  from .serialisation import transformers as transformers

# NOTE: update this to sys.modules[__name__] once mypy_extensions can recognize __spec__
__lazy = openllm_core.utils.LazyModule(__name__, globals()['__file__'], _import_structure, extra_objects={'COMPILED': COMPILED})
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__
