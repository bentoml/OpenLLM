# mypy: disable-error-code="assignment,has-type,no-redef"
# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenLLM.

An open platform for operating large language models in production. Fine-tune, serve,
deploy, and monitor any LLMs with ease.

* Built-in support for StableLM, Llama 2, Dolly, Flan-T5, Vicuna
* Option to bring your own fine-tuned LLMs
* Online Serving with HTTP, gRPC, SSE(coming soon) or custom API
* Native integration with BentoML and LangChain for custom LLM apps
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
import warnings

from . import utils as utils

COMPILED = Path(__file__).suffix in (".pyd", ".so")

if utils.DEBUG:
  utils.set_debug_mode(True)
  utils.set_quiet_mode(False)
  logging.basicConfig(level=logging.NOTSET)
else:
  # configuration for bitsandbytes before import
  os.environ["BITSANDBYTES_NOWELCOME"] = os.environ.get("BITSANDBYTES_NOWELCOME", "1")
  # NOTE: The following warnings from bitsandbytes, and probably not that important for users to see when DEBUG is False
  warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")
  warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization")
  warnings.filterwarnings("ignore", message="The installed version of bitsandbytes was compiled without GPU support.")
  # NOTE: ignore the following warning from ghapi as it is not important for users
  warnings.filterwarnings("ignore", message="Neither GITHUB_TOKEN nor GITHUB_JWT_TOKEN found: running as unauthenticated")

from . import bundle as bundle
from . import cli as cli
from . import client as client
from . import exceptions as exceptions
from . import models as models
from . import playground as playground
from . import serialisation as serialisation
from . import testing as testing
from ._configuration import GenerationConfig as GenerationConfig
from ._configuration import LLMConfig as LLMConfig
from ._configuration import SamplingParams as SamplingParams
from ._generation import LogitsProcessorList as LogitsProcessorList
from ._generation import StopOnTokens as StopOnTokens
from ._generation import StoppingCriteriaList as StoppingCriteriaList
from ._generation import StopSequenceCriteria as StopSequenceCriteria
from ._generation import prepare_logits_processor as prepare_logits_processor
from ._llm import LLM as LLM
from ._llm import LLMEmbeddings as LLMEmbeddings
from ._llm import LLMRunnable as LLMRunnable
from ._llm import LLMRunner as LLMRunner
from ._llm import Runner as Runner
from ._quantisation import infer_quantisation_config as infer_quantisation_config
from ._schema import EmbeddingsOutput as EmbeddingsOutput
from ._schema import GenerationInput as GenerationInput
from ._schema import GenerationOutput as GenerationOutput
from ._schema import HfAgentInput as HfAgentInput
from ._schema import MetadataOutput as MetadataOutput
from ._schema import unmarshal_vllm_outputs as unmarshal_vllm_outputs
from .cli._sdk import build as build
from .cli._sdk import import_model as import_model
from .cli._sdk import list_models as list_models
from .cli._sdk import start as start
from .cli._sdk import start_grpc as start_grpc
from .models.auto import CONFIG_MAPPING as CONFIG_MAPPING
from .models.auto import MODEL_FLAX_MAPPING_NAMES as MODEL_FLAX_MAPPING_NAMES
from .models.auto import MODEL_MAPPING_NAMES as MODEL_MAPPING_NAMES
from .models.auto import MODEL_TF_MAPPING_NAMES as MODEL_TF_MAPPING_NAMES
from .models.auto import MODEL_VLLM_MAPPING_NAMES as MODEL_VLLM_MAPPING_NAMES
from .models.auto import AutoConfig as AutoConfig
from .serialisation import ggml as ggml
from .serialisation import transformers as transformers
from .utils import infer_auto_class as infer_auto_class

# NOTE: torch and cpm_kernels
try:
  if not (utils.is_torch_available() and utils.is_cpm_kernels_available()): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError: from openllm.utils.dummy_pt_and_cpm_kernels_objects import *
else:
  from .models.baichuan import Baichuan as Baichuan
  from .models.chatglm import ChatGLM as ChatGLM

try:
  if not (utils.is_torch_available() and utils.is_einops_available()): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError: from openllm.utils.dummy_pt_and_einops_objects import *
else: from .models.falcon import Falcon as Falcon

try:
  if not (utils.is_torch_available() and utils.is_triton_available()): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError: from openllm.utils.dummy_pt_and_triton_objects import *
else: from .models.mpt import MPT as MPT

try:
  if not utils.is_torch_available(): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError: from openllm.utils.dummy_pt_objects import *
else:
  from .models.auto import MODEL_MAPPING as MODEL_MAPPING
  from .models.auto import AutoLLM as AutoLLM
  from .models.dolly_v2 import DollyV2 as DollyV2
  from .models.flan_t5 import FlanT5 as FlanT5
  from .models.gpt_neox import GPTNeoX as GPTNeoX
  from .models.llama import Llama as Llama
  from .models.opt import OPT as OPT
  from .models.stablelm import StableLM as StableLM
  from .models.starcoder import StarCoder as StarCoder

try:
  if not utils.is_vllm_available(): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError: from openllm.utils.dummy_vllm_objects import *
else:
  from .models.auto import MODEL_VLLM_MAPPING as MODEL_VLLM_MAPPING
  from .models.auto import AutoVLLM as AutoVLLM
  from .models.baichuan import VLLMBaichuan as VLLMBaichuan
  from .models.llama import VLLMLlama as VLLMLlama
  from .models.opt import VLLMOPT as VLLMOPT

try:
  if not utils.is_flax_available(): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError: from openllm.utils.dummy_flax_objects import *
else:
  from .models.auto import MODEL_FLAX_MAPPING as MODEL_FLAX_MAPPING
  from .models.auto import AutoFlaxLLM as AutoFlaxLLM
  from .models.flan_t5 import FlaxFlanT5 as FlaxFlanT5
  from .models.opt import FlaxOPT as FlaxOPT

try:
  if not utils.is_tf_available(): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError: from openllm.utils.dummy_tf_objects import *
else:
  from .models.auto import MODEL_TF_MAPPING as MODEL_TF_MAPPING
  from .models.auto import AutoTFLLM as AutoTFLLM
  from .models.flan_t5 import TFFlanT5 as TFFlanT5
  from .models.opt import TFOPT as TFOPT

# NOTE: Make sure to always keep __all__ at the bottom for ./tools/update-init-import.py, and DO NOT MODIFIED THE LINE BELOW
__all__=["AutoConfig","AutoFlaxLLM","AutoLLM","AutoTFLLM","AutoVLLM","Baichuan","COMPILED","CONFIG_MAPPING","ChatGLM","DollyV2","DummyMetaclass","EmbeddingsOutput","Falcon","FlanT5","FlaxFlanT5","FlaxOPT","GPTNeoX","GenerationConfig","GenerationInput","GenerationOutput","HfAgentInput","LLM","LLMConfig","LLMEmbeddings","LLMRunnable","LLMRunner","Llama","LogitsProcessorList","MODEL_FLAX_MAPPING","MODEL_FLAX_MAPPING_NAMES","MODEL_MAPPING","MODEL_MAPPING_NAMES","MODEL_TF_MAPPING","MODEL_TF_MAPPING_NAMES","MODEL_VLLM_MAPPING","MODEL_VLLM_MAPPING_NAMES","MPT","MetadataOutput","OPT","Runner","SamplingParams","StableLM","StarCoder","StopOnTokens","StopSequenceCriteria","StoppingCriteriaList","TFFlanT5","TFOPT","VLLMBaichuan","VLLMDollyV2","VLLMGPTNeoX","VLLMLlama","VLLMMPT","VLLMOPT","VLLMStableLM","VLLMStarCoder","annotations","build","bundle","cli","client","exceptions","ggml","import_model","infer_auto_class","infer_quantisation_config","list_models","models","playground","prepare_logits_processor","require_backends","serialisation","start","start_grpc","t","testing","transformers","unmarshal_vllm_outputs","utils","COMPILED"]
