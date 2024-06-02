"""OpenLLM.
===========

An open platform for operating large language models in production.
Fine-tune, serve, deploy, and monitor any LLMs with ease.

* Built-in support for Mistral, Llama 2, Yi, StableLM, Dolly, Flan-T5, Vicuna
* Option to bring your own fine-tuned LLMs
* Online Serving with HTTP, gRPC, SSE or custom API
* Native integration with BentoML, LangChain, OpenAI compatible endpoints, LlamaIndex for custom LLM apps
"""

# fmt: off
# update-config-stubs.py: import stubs start
from openllm_client import AsyncHTTPClient as AsyncHTTPClient, HTTPClient as HTTPClient
from openlm_core.config import CONFIG_MAPPING as CONFIG_MAPPING, CONFIG_MAPPING_NAMES as CONFIG_MAPPING_NAMES, AutoConfig as AutoConfig, BaichuanConfig as BaichuanConfig, ChatGLMConfig as ChatGLMConfig, CohereConfig as CohereConfig, DbrxConfig as DbrxConfig, DollyV2Config as DollyV2Config, FalconConfig as FalconConfig, GemmaConfig as GemmaConfig, GPTNeoXConfig as GPTNeoXConfig, LlamaConfig as LlamaConfig, MistralConfig as MistralConfig, MixtralConfig as MixtralConfig, MPTConfig as MPTConfig, OPTConfig as OPTConfig, PhiConfig as PhiConfig, QwenConfig as QwenConfig, StableLMConfig as StableLMConfig, StarCoderConfig as StarCoderConfig, YiConfig as YiConfig
from openllm_core._configuration import GenerationConfig as GenerationConfig, LLMConfig as LLMConfig
from openllm_core._schemas import GenerationInput as GenerationInput, GenerationOutput as GenerationOutput, MetadataOutput as MetadataOutput, MessageParam as MessageParam
from openllm_core.utils import api as api
# update-config-stubs.py: import stubs stop
# fmt: on

from . import (
  bundle as bundle,
  client as client,
  exceptions as exceptions,
  serialisation as serialisation,
  utils as utils,
)
from .serialisation import ggml as ggml, transformers as transformers, vllm as vllm
from ._deprecated import Runner as Runner
from ._runners import runner as runner
from ._quantisation import infer_quantisation_config as infer_quantisation_config
from ._strategies import CascadingResourceStrategy as CascadingResourceStrategy, get_resource as get_resource
from _openllm_tiny import LLM as LLM

COMPILED: bool = ...
