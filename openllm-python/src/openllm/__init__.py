"""OpenLLM.

An open platform for operating large language models in production. Fine-tune, serve,
deploy, and monitor any LLMs with ease.

* Built-in support for StableLM, Llama 2, Dolly, Flan-T5, Vicuna
* Option to bring your own fine-tuned LLMs
* Online Serving with HTTP, gRPC, SSE(coming soon) or custom API
* Native integration with BentoML and LangChain for custom LLM apps
"""
from __future__ import annotations
import logging as _logging, os as _os, typing as _t, warnings as _warnings, openllm_core
from pathlib import Path as _Path
from . import exceptions as exceptions, utils as utils

from openllm_core._configuration import GenerationConfig as GenerationConfig, LLMConfig as LLMConfig, SamplingParams as SamplingParams
from openllm_core._strategies import CascadingResourceStrategy as CascadingResourceStrategy, get_resource as get_resource
from openllm_core._schema import EmbeddingsOutput as EmbeddingsOutput, GenerationInput as GenerationInput, GenerationOutput as GenerationOutput, HfAgentInput as HfAgentInput, MetadataOutput as MetadataOutput, unmarshal_vllm_outputs as unmarshal_vllm_outputs
from openllm_core.config import AutoConfig as AutoConfig, CONFIG_MAPPING as CONFIG_MAPPING, CONFIG_MAPPING_NAMES as CONFIG_MAPPING_NAMES, BaichuanConfig as BaichuanConfig, ChatGLMConfig as ChatGLMConfig, DollyV2Config as DollyV2Config, FalconConfig as FalconConfig, FlanT5Config as FlanT5Config, GPTNeoXConfig as GPTNeoXConfig, LlamaConfig as LlamaConfig, MPTConfig as MPTConfig, OPTConfig as OPTConfig, StableLMConfig as StableLMConfig, StarCoderConfig as StarCoderConfig

if openllm_core.utils.DEBUG:
  openllm_core.utils.set_debug_mode(True)
  openllm_core.utils.set_quiet_mode(False)
  _logging.basicConfig(level=_logging.NOTSET)
else:
  # configuration for bitsandbytes before import
  _os.environ["BITSANDBYTES_NOWELCOME"] = _os.environ.get("BITSANDBYTES_NOWELCOME", "1")
  # NOTE: The following warnings from bitsandbytes, and probably not that important for users to see when DEBUG is False
  _warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.float32 to float16 during quantization")
  _warnings.filterwarnings("ignore", message="MatMul8bitLt: inputs will be cast from torch.bfloat16 to float16 during quantization")
  _warnings.filterwarnings("ignore", message="The installed version of bitsandbytes was compiled without GPU support.")
  # NOTE: ignore the following warning from ghapi as it is not important for users
  _warnings.filterwarnings("ignore", message="Neither GITHUB_TOKEN nor GITHUB_JWT_TOKEN found: running as unauthenticated")

_import_structure: dict[str, list[str]] = {
    "exceptions": [],
    "models": [],
    "client": [],
    "bundle": [],
    "playground": [],
    "testing": [],
    "utils": ["infer_auto_class"],
    "serialisation": ["ggml", "transformers"],
    "cli._sdk": ["start", "start_grpc", "build", "import_model", "list_models"],
    "_quantisation": ["infer_quantisation_config"],
    "_embeddings": ["GenericEmbeddingRunnable"],
    "_llm": ["LLM", "Runner", "LLMRunner", "LLMRunnable", "EmbeddingsOutput"],
    "_generation": ["StopSequenceCriteria", "StopOnTokens", "LogitsProcessorList", "StoppingCriteriaList", "prepare_logits_processor"],
    "models.auto": ["MODEL_MAPPING_NAMES", "MODEL_FLAX_MAPPING_NAMES", "MODEL_TF_MAPPING_NAMES", "MODEL_VLLM_MAPPING_NAMES"],
    "models.chatglm": [],
    "models.baichuan": [],
    "models.dolly_v2": [],
    "models.falcon": [],
    "models.flan_t5": [],
    "models.gpt_neox": [],
    "models.llama": [],
    "models.mpt": [],
    "models.opt": [],
    "models.stablelm": [],
    "models.starcoder": []
}
COMPILED = _Path(__file__).suffix in (".pyd", ".so")

if _t.TYPE_CHECKING:
  from . import bundle as bundle, cli as cli, client as client, models as models, playground as playground, serialisation as serialisation, testing as testing
  from ._generation import LogitsProcessorList as LogitsProcessorList, StopOnTokens as StopOnTokens, StoppingCriteriaList as StoppingCriteriaList, StopSequenceCriteria as StopSequenceCriteria, prepare_logits_processor as prepare_logits_processor
  from ._llm import LLM as LLM, EmbeddingsOutput as EmbeddingsOutput, LLMRunnable as LLMRunnable, LLMRunner as LLMRunner, Runner as Runner
  from ._quantisation import infer_quantisation_config as infer_quantisation_config
  from ._embeddings import GenericEmbeddingRunnable as GenericEmbeddingRunnable
  from .cli._sdk import build as build, import_model as import_model, list_models as list_models, start as start, start_grpc as start_grpc
  from .models.auto import MODEL_FLAX_MAPPING_NAMES as MODEL_FLAX_MAPPING_NAMES, MODEL_MAPPING_NAMES as MODEL_MAPPING_NAMES, MODEL_TF_MAPPING_NAMES as MODEL_TF_MAPPING_NAMES, MODEL_VLLM_MAPPING_NAMES as MODEL_VLLM_MAPPING_NAMES
  from .serialisation import ggml as ggml, transformers as transformers
  from .utils import infer_auto_class as infer_auto_class

try:
  if not (openllm_core.utils.is_torch_available() and openllm_core.utils.is_cpm_kernels_available()):
    raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  _import_structure["utils.dummy_pt_objects"] = ["ChatGLM", "Baichuan"]
else:
  _import_structure["models.chatglm"].extend(["ChatGLM"])
  _import_structure["models.baichuan"].extend(["Baichuan"])
  if _t.TYPE_CHECKING:
    from .models.baichuan import Baichuan as Baichuan
    from .models.chatglm import ChatGLM as ChatGLM
try:
  if not (openllm_core.utils.is_torch_available() and openllm_core.utils.is_triton_available()):
    raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  if "utils.dummy_pt_objects" in _import_structure: _import_structure["utils.dummy_pt_objects"].extend(["MPT"])
  else: _import_structure["utils.dummy_pt_objects"] = ["MPT"]
else:
  _import_structure["models.mpt"].extend(["MPT"])
  if _t.TYPE_CHECKING: from .models.mpt import MPT as MPT
try:
  if not (openllm_core.utils.is_torch_available() and openllm_core.utils.is_einops_available()):
    raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  if "utils.dummy_pt_objects" in _import_structure: _import_structure["utils.dummy_pt_objects"].extend(["Falcon"])
  else: _import_structure["utils.dummy_pt_objects"] = ["Falcon"]
else:
  _import_structure["models.falcon"].extend(["Falcon"])
  if _t.TYPE_CHECKING: from .models.falcon import Falcon as Falcon

try:
  if not openllm_core.utils.is_torch_available(): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  _import_structure["utils.dummy_pt_objects"] = [
      name for name in dir(utils.dummy_pt_objects) if not name.startswith("_") and name not in ("ChatGLM", "Baichuan", "MPT", "Falcon", "annotations")
  ]
else:
  _import_structure["models.flan_t5"].extend(["FlanT5"])
  _import_structure["models.dolly_v2"].extend(["DollyV2"])
  _import_structure["models.starcoder"].extend(["StarCoder"])
  _import_structure["models.stablelm"].extend(["StableLM"])
  _import_structure["models.opt"].extend(["OPT"])
  _import_structure["models.gpt_neox"].extend(["GPTNeoX"])
  _import_structure["models.llama"].extend(["Llama"])
  _import_structure["models.auto"].extend(["AutoLLM", "MODEL_MAPPING"])
  if _t.TYPE_CHECKING:
    from .models.auto import MODEL_MAPPING as MODEL_MAPPING, AutoLLM as AutoLLM
    from .models.dolly_v2 import DollyV2 as DollyV2
    from .models.flan_t5 import FlanT5 as FlanT5
    from .models.gpt_neox import GPTNeoX as GPTNeoX
    from .models.llama import Llama as Llama
    from .models.opt import OPT as OPT
    from .models.stablelm import StableLM as StableLM
    from .models.starcoder import StarCoder as StarCoder
try:
  if not openllm_core.utils.is_vllm_available(): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  _import_structure["utils.dummy_vllm_objects"] = [name for name in dir(utils.dummy_vllm_objects) if not name.startswith("_") and name not in ("annotations",)]
else:
  _import_structure["models.baichuan"].extend(["VLLMBaichuan"])
  _import_structure["models.llama"].extend(["VLLMLlama"])
  _import_structure["models.opt"].extend(["VLLMOPT"])
  _import_structure["models.dolly_v2"].extend(["VLLMDollyV2"])
  _import_structure["models.falcon"].extend(["VLLMFalcon"])
  _import_structure["models.gpt_neox"].extend(["VLLMGPTNeoX"])
  _import_structure["models.mpt"].extend(["VLLMMPT"])
  _import_structure["models.stablelm"].extend(["VLLMStableLM"])
  _import_structure["models.starcoder"].extend(["VLLMStarCoder"])
  _import_structure["models.auto"].extend(["AutoVLLM", "MODEL_VLLM_MAPPING"])
  if _t.TYPE_CHECKING:
    from .models.auto import MODEL_VLLM_MAPPING as MODEL_VLLM_MAPPING, AutoVLLM as AutoVLLM
    from .models.baichuan import VLLMBaichuan as VLLMBaichuan
    from .models.dolly_v2 import VLLMDollyV2 as VLLMDollyV2
    from .models.gpt_neox import VLLMGPTNeoX as VLLMGPTNeoX
    from .models.falcon import VLLMFalcon as VLLMFalcon
    from .models.llama import VLLMLlama as VLLMLlama
    from .models.mpt import VLLMMPT as VLLMMPT
    from .models.opt import VLLMOPT as VLLMOPT
    from .models.stablelm import VLLMStableLM as VLLMStableLM
    from .models.starcoder import VLLMStarCoder as VLLMStarCoder
try:
  if not openllm_core.utils.is_flax_available(): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  _import_structure["utils.dummy_flax_objects"] = [name for name in dir(utils.dummy_flax_objects) if not name.startswith("_") and name not in ("annotations",)]
else:
  _import_structure["models.flan_t5"].extend(["FlaxFlanT5"])
  _import_structure["models.opt"].extend(["FlaxOPT"])
  _import_structure["models.auto"].extend(["AutoFlaxLLM", "MODEL_FLAX_MAPPING"])
  if _t.TYPE_CHECKING:
    from .models.auto import MODEL_FLAX_MAPPING as MODEL_FLAX_MAPPING, AutoFlaxLLM as AutoFlaxLLM
    from .models.flan_t5 import FlaxFlanT5 as FlaxFlanT5
    from .models.opt import FlaxOPT as FlaxOPT
try:
  if not openllm_core.utils.is_tf_available(): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  _import_structure["utils.dummy_tf_objects"] = [name for name in dir(utils.dummy_tf_objects) if not name.startswith("_") and name not in ("annotations",)]
else:
  _import_structure["models.flan_t5"].extend(["TFFlanT5"])
  _import_structure["models.opt"].extend(["TFOPT"])
  _import_structure["models.auto"].extend(["AutoTFLLM", "MODEL_TF_MAPPING"])
  if _t.TYPE_CHECKING:
    from .models.auto import MODEL_TF_MAPPING as MODEL_TF_MAPPING, AutoTFLLM as AutoTFLLM
    from .models.flan_t5 import TFFlanT5 as TFFlanT5
    from .models.opt import TFOPT as TFOPT

# NOTE: update this to sys.modules[__name__] once mypy_extensions can recognize __spec__
__lazy = openllm_core.utils.LazyModule(__name__, globals()["__file__"], _import_structure, extra_objects={"COMPILED": COMPILED, "__openllm_migration__": {"LLMEmbeddings": "EmbeddingsOutput"}})
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__
