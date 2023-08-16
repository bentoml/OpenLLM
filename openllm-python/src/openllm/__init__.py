"""OpenLLM.

An open platform for operating large language models in production. Fine-tune, serve,
deploy, and monitor any LLMs with ease.

* Built-in support for StableLM, Llama 2, Dolly, Flan-T5, Vicuna
* Option to bring your own fine-tuned LLMs
* Online Serving with HTTP, gRPC, SSE(coming soon) or custom API
* Native integration with BentoML and LangChain for custom LLM apps
"""
from __future__ import annotations
import logging as _logging, os as _os, typing as _t, warnings as _warnings
from pathlib import Path as _Path
from . import exceptions as exceptions, utils as utils

if utils.DEBUG:
  utils.set_debug_mode(True)
  utils.set_quiet_mode(False)
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
    "exceptions": [], "models": [], "client": [], "bundle": [], "playground": [], "testing": [], "utils": ["infer_auto_class"], "serialisation": ["ggml", "transformers"], "cli._sdk": ["start", "start_grpc", "build", "import_model", "list_models"], "_llm": ["LLM", "Runner", "LLMRunner", "LLMRunnable", "LLMEmbeddings"], "_configuration": ["LLMConfig", "GenerationConfig", "SamplingParams"], "_generation": ["StopSequenceCriteria", "StopOnTokens", "LogitsProcessorList", "StoppingCriteriaList", "prepare_logits_processor"], "_quantisation": ["infer_quantisation_config"], "_schema": ["GenerationInput", "GenerationOutput", "MetadataOutput", "EmbeddingsOutput", "unmarshal_vllm_outputs", "HfAgentInput"],
    "models.auto": ["AutoConfig", "CONFIG_MAPPING", "MODEL_MAPPING_NAMES", "MODEL_FLAX_MAPPING_NAMES", "MODEL_TF_MAPPING_NAMES", "MODEL_VLLM_MAPPING_NAMES"], "models.chatglm": ["ChatGLMConfig"], "models.baichuan": ["BaichuanConfig"], "models.dolly_v2": ["DollyV2Config"], "models.falcon": ["FalconConfig"], "models.flan_t5": ["FlanT5Config"], "models.gpt_neox": ["GPTNeoXConfig"], "models.llama": ["LlamaConfig"], "models.mpt": ["MPTConfig"], "models.opt": ["OPTConfig"], "models.stablelm": ["StableLMConfig"], "models.starcoder": ["StarCoderConfig"]
}
COMPILED = _Path(__file__).suffix in (".pyd", ".so")

if _t.TYPE_CHECKING:
  from . import bundle as bundle, cli as cli, client as client, models as models, playground as playground, serialisation as serialisation, testing as testing
  from ._configuration import GenerationConfig as GenerationConfig, LLMConfig as LLMConfig, SamplingParams as SamplingParams
  from ._generation import LogitsProcessorList as LogitsProcessorList, StopOnTokens as StopOnTokens, StoppingCriteriaList as StoppingCriteriaList, StopSequenceCriteria as StopSequenceCriteria, prepare_logits_processor as prepare_logits_processor
  from ._llm import LLM as LLM, LLMEmbeddings as LLMEmbeddings, LLMRunnable as LLMRunnable, LLMRunner as LLMRunner, Runner as Runner
  from ._quantisation import infer_quantisation_config as infer_quantisation_config
  from ._schema import EmbeddingsOutput as EmbeddingsOutput, GenerationInput as GenerationInput, GenerationOutput as GenerationOutput, HfAgentInput as HfAgentInput, MetadataOutput as MetadataOutput, unmarshal_vllm_outputs as unmarshal_vllm_outputs
  from .cli._sdk import build as build, import_model as import_model, list_models as list_models, start as start, start_grpc as start_grpc
  from .models.auto import CONFIG_MAPPING as CONFIG_MAPPING, MODEL_FLAX_MAPPING_NAMES as MODEL_FLAX_MAPPING_NAMES, MODEL_MAPPING_NAMES as MODEL_MAPPING_NAMES, MODEL_TF_MAPPING_NAMES as MODEL_TF_MAPPING_NAMES, MODEL_VLLM_MAPPING_NAMES as MODEL_VLLM_MAPPING_NAMES, AutoConfig as AutoConfig
  from .models.baichuan import BaichuanConfig as BaichuanConfig
  from .models.chatglm import ChatGLMConfig as ChatGLMConfig
  from .models.dolly_v2 import DollyV2Config as DollyV2Config
  from .models.falcon import FalconConfig as FalconConfig
  from .models.flan_t5 import FlanT5Config as FlanT5Config
  from .models.gpt_neox import GPTNeoXConfig as GPTNeoXConfig
  from .models.llama import LlamaConfig as LlamaConfig
  from .models.mpt import MPTConfig as MPTConfig
  from .models.opt import OPTConfig as OPTConfig
  from .models.stablelm import StableLMConfig as StableLMConfig
  from .models.starcoder import StarCoderConfig as StarCoderConfig
  from .serialisation import ggml as ggml, transformers as transformers
  from openllm.utils import infer_auto_class as infer_auto_class

try:
  if not (utils.is_torch_available() and utils.is_cpm_kernels_available()): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  _import_structure["utils.dummy_pt_objects"] = ["ChatGLM", "Baichuan"]
else:
  _import_structure["models.chatglm"].extend(["ChatGLM"])
  _import_structure["models.baichuan"].extend(["Baichuan"])
  if _t.TYPE_CHECKING:
    from .models.baichuan import Baichuan as Baichuan
    from .models.chatglm import ChatGLM as ChatGLM
try:
  if not (utils.is_torch_available() and utils.is_triton_available()): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  if "utils.dummy_pt_objects" in _import_structure: _import_structure["utils.dummy_pt_objects"].extend(["MPT"])
  else: _import_structure["utils.dummy_pt_objects"] = ["MPT"]
else:
  _import_structure["models.mpt"].extend(["MPT"])
  if _t.TYPE_CHECKING: from .models.mpt import MPT as MPT
try:
  if not (utils.is_torch_available() and utils.is_einops_available()): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  if "utils.dummy_pt_objects" in _import_structure: _import_structure["utils.dummy_pt_objects"].extend(["Falcon"])
  else: _import_structure["utils.dummy_pt_objects"] = ["Falcon"]
else:
  _import_structure["models.falcon"].extend(["Falcon"])
  if _t.TYPE_CHECKING: from .models.falcon import Falcon as Falcon

try:
  if not utils.is_torch_available(): raise exceptions.MissingDependencyError
except exceptions.MissingDependencyError:
  _import_structure["utils.dummy_pt_objects"] = [name for name in dir(utils.dummy_pt_objects) if not name.startswith("_") and name not in ("ChatGLM", "Baichuan", "MPT", "Falcon", "annotations")]
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
  if not utils.is_vllm_available(): raise exceptions.MissingDependencyError
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
  if not utils.is_flax_available(): raise exceptions.MissingDependencyError
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
  if not utils.is_tf_available(): raise exceptions.MissingDependencyError
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
__lazy = utils.LazyModule(__name__, _os.path.abspath("__file__"), _import_structure, extra_objects={"COMPILED": COMPILED})
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__
