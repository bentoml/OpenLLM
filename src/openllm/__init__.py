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
import sys
import typing as t
import warnings

from . import utils as utils
from .exceptions import MissingDependencyError

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

_import_structure: dict[str, list[str]] = {
    "_llm": ["LLM", "Runner", "LLMRunner", "LLMRunnable", "LLMEmbeddings"], "_configuration": ["LLMConfig", "GenerationConfig", "SamplingParams"], "_schema": ["GenerationInput", "GenerationOutput", "MetadataOutput", "EmbeddingsOutput", "unmarshal_vllm_outputs", "HfAgentInput"], "_generation": [
        "StopSequenceCriteria", "StopOnTokens", "LogitsProcessorList", "StoppingCriteriaList", "prepare_logits_processor"
    ], "_quantisation": ["infer_quantisation_config"], "exceptions": [], "utils": ["infer_auto_class"], "models": [], "client": [], "bundle": [], "playground": [], "testing": [], "serialisation": ["ggml", "transformers"], "cli.entrypoint": ["start", "start_grpc", "build", "import_model", "list_models"],
    # NOTE: models
    "models.auto": ["AutoConfig", "CONFIG_MAPPING", "MODEL_MAPPING_NAMES", "MODEL_FLAX_MAPPING_NAMES", "MODEL_TF_MAPPING_NAMES", "MODEL_VLLM_MAPPING_NAMES"], "models.chatglm": ["ChatGLMConfig"], "models.baichuan": ["BaichuanConfig"], "models.dolly_v2": ["DollyV2Config"], "models.falcon": ["FalconConfig"], "models.flan_t5": ["FlanT5Config"], "models.gpt_neox": ["GPTNeoXConfig"],
    "models.llama": ["LlamaConfig"], "models.mpt": ["MPTConfig"], "models.opt": ["OPTConfig"], "models.stablelm": ["StableLMConfig"], "models.starcoder": ["StarCoderConfig"],
}

# NOTE: torch and cpm_kernels
try:
  if not (utils.is_torch_available() and utils.is_cpm_kernels_available()): raise MissingDependencyError
except MissingDependencyError:
  from .utils import dummy_pt_and_cpm_kernels_objects
  _import_structure["utils.dummy_pt_and_cpm_kernels_objects"] = [name for name in dir(dummy_pt_and_cpm_kernels_objects) if not name.startswith("_")]
else:
  _import_structure["models.chatglm"].extend(["ChatGLM"])
  _import_structure["models.baichuan"].extend(["Baichuan"])

try:
  if not (utils.is_torch_available() and utils.is_einops_available()): raise MissingDependencyError
except MissingDependencyError:
  from .utils import dummy_pt_and_einops_objects
  _import_structure["utils.dummy_pt_and_einops_objects"] = [name for name in dir(dummy_pt_and_einops_objects) if not name.startswith("_")]
else:
  _import_structure["models.falcon"].extend(["Falcon"])

try:
  if not (utils.is_torch_available() and utils.is_triton_available()): raise MissingDependencyError
except MissingDependencyError:
  from .utils import dummy_pt_and_triton_objects
  _import_structure["utils.dummy_pt_and_triton_objects"] = [name for name in dir(dummy_pt_and_triton_objects) if not name.startswith("_")]
else:
  _import_structure["models.mpt"].extend(["MPT"])

try:
  if not utils.is_torch_available(): raise MissingDependencyError
except MissingDependencyError:
  from .utils import dummy_pt_objects
  _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
  _import_structure["models.flan_t5"].extend(["FlanT5"])
  _import_structure["models.dolly_v2"].extend(["DollyV2"])
  _import_structure["models.starcoder"].extend(["StarCoder"])
  _import_structure["models.stablelm"].extend(["StableLM"])
  _import_structure["models.opt"].extend(["OPT"])
  _import_structure["models.gpt_neox"].extend(["GPTNeoX"])
  _import_structure["models.llama"].extend(["Llama"])
  _import_structure["models.auto"].extend(["AutoLLM", "MODEL_MAPPING"])

try:
  if not utils.is_vllm_available(): raise MissingDependencyError
except MissingDependencyError:
  from .utils import dummy_vllm_objects
  _import_structure["utils.dummy_vllm_objects"] = [name for name in dir(dummy_vllm_objects) if not name.startswith("_")]
else:
  _import_structure["models.baichuan"].extend(["VLLMBaichuan"])
  _import_structure["models.llama"].extend(["VLLMLlama"])
  _import_structure["models.opt"].extend(["VLLMOPT"])
  _import_structure["models.auto"].extend(["AutoVLLM", "MODEL_VLLM_MAPPING"])

try:
  if not utils.is_flax_available(): raise MissingDependencyError
except MissingDependencyError:
  from .utils import dummy_flax_objects
  _import_structure["utils.dummy_flax_objects"] = [name for name in dir(dummy_flax_objects) if not name.startswith("_")]
else:
  _import_structure["models.flan_t5"].extend(["FlaxFlanT5"])
  _import_structure["models.opt"].extend(["FlaxOPT"])
  _import_structure["models.auto"].extend(["AutoFlaxLLM", "MODEL_FLAX_MAPPING"])

try:
  if not utils.is_tf_available(): raise MissingDependencyError
except MissingDependencyError:
  from .utils import dummy_tf_objects
  _import_structure["utils.dummy_tf_objects"] = [name for name in dir(dummy_tf_objects) if not name.startswith("_")]
else:
  _import_structure["models.flan_t5"].extend(["TFFlanT5"])
  _import_structure["models.opt"].extend(["TFOPT"])
  _import_structure["models.auto"].extend(["AutoTFLLM", "MODEL_TF_MAPPING"])

# declaration for OpenLLM-related modules
if t.TYPE_CHECKING:
  from . import bundle as bundle
  from . import cli as cli
  from . import client as client
  from . import exceptions as exceptions
  from . import models as models
  from . import playground as playground
  from . import serialisation as serialisation
  from . import testing as testing
  from ._configuration import GenerationConfig as GenerationConfig

  # Specific types import
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
  from .cli.entrypoint import build as build
  from .cli.entrypoint import import_model as import_model
  from .cli.entrypoint import list_models as list_models
  from .cli.entrypoint import start as start
  from .cli.entrypoint import start_grpc as start_grpc
  from .models.auto import CONFIG_MAPPING as CONFIG_MAPPING

  # Flax imports
  from .models.auto import MODEL_FLAX_MAPPING as MODEL_FLAX_MAPPING
  from .models.auto import MODEL_FLAX_MAPPING_NAMES as MODEL_FLAX_MAPPING_NAMES
  from .models.auto import MODEL_MAPPING as MODEL_MAPPING
  from .models.auto import MODEL_MAPPING_NAMES as MODEL_MAPPING_NAMES

  # Tensorflow imports
  from .models.auto import MODEL_TF_MAPPING as MODEL_TF_MAPPING
  from .models.auto import MODEL_TF_MAPPING_NAMES as MODEL_TF_MAPPING_NAMES

  # vLLM imports
  from .models.auto import MODEL_VLLM_MAPPING as MODEL_VLLM_MAPPING
  from .models.auto import MODEL_VLLM_MAPPING_NAMES as MODEL_VLLM_MAPPING_NAMES
  from .models.auto import AutoConfig as AutoConfig
  from .models.auto import AutoFlaxLLM as AutoFlaxLLM
  from .models.auto import AutoLLM as AutoLLM
  from .models.auto import AutoTFLLM as AutoTFLLM
  from .models.auto import AutoVLLM as AutoVLLM

  # PyTorch imports
  from .models.baichuan import Baichuan as Baichuan
  from .models.baichuan import BaichuanConfig as BaichuanConfig
  from .models.baichuan import VLLMBaichuan as VLLMBaichuan
  from .models.chatglm import ChatGLM as ChatGLM
  from .models.chatglm import ChatGLMConfig as ChatGLMConfig
  from .models.dolly_v2 import DollyV2 as DollyV2
  from .models.dolly_v2 import DollyV2Config as DollyV2Config
  from .models.dolly_v2 import VLLMDollyV2 as VLLMDollyV2
  from .models.falcon import Falcon as Falcon
  from .models.falcon import FalconConfig as FalconConfig
  from .models.flan_t5 import FlanT5 as FlanT5
  from .models.flan_t5 import FlanT5Config as FlanT5Config
  from .models.flan_t5 import FlaxFlanT5 as FlaxFlanT5
  from .models.flan_t5 import TFFlanT5 as TFFlanT5
  from .models.gpt_neox import GPTNeoX as GPTNeoX
  from .models.gpt_neox import GPTNeoXConfig as GPTNeoXConfig
  from .models.gpt_neox import VLLMGPTNeoX as VLLMGPTNeoX
  from .models.llama import Llama as Llama
  from .models.llama import LlamaConfig as LlamaConfig
  from .models.llama import VLLMLlama as VLLMLlama
  from .models.mpt import MPT as MPT
  from .models.mpt import VLLMMPT as VLLMMPT
  from .models.mpt import MPTConfig as MPTConfig
  from .models.opt import OPT as OPT
  from .models.opt import TFOPT as TFOPT
  from .models.opt import VLLMOPT as VLLMOPT
  from .models.opt import FlaxOPT as FlaxOPT
  from .models.opt import OPTConfig as OPTConfig
  from .models.stablelm import StableLM as StableLM
  from .models.stablelm import StableLMConfig as StableLMConfig
  from .models.stablelm import VLLMStableLM as VLLMStableLM
  from .models.starcoder import StarCoder as StarCoder
  from .models.starcoder import StarCoderConfig as StarCoderConfig
  from .models.starcoder import VLLMStarCoder as VLLMStarCoder
  from .serialisation import ggml as ggml
  from .serialisation import transformers as transformers
  from .utils import infer_auto_class as infer_auto_class
else:
  sys.modules[__name__] = utils.LazyModule(
      __name__,
      globals()["__file__"], _import_structure, module_spec=__spec__, doc=__doc__, extra_objects={
          # The below is a special mapping that allows openllm to be used as a dictionary.
          # This is purely for convenience sake, and should not be used in performance critcal
          # code. This is also not considered as a public API.
          "__openllm_special__": {"flax": "AutoFlaxLLM", "tf": "AutoTFLLM", "pt": "AutoLLM", "vllm": "AutoVLLM"},
      }
  )
