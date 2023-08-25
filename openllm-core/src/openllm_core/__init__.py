from __future__ import annotations

from . import exceptions as exceptions, utils as utils
from ._configuration import GenerationConfig as GenerationConfig, LLMConfig as LLMConfig, SamplingParams as SamplingParams
from ._schema import EmbeddingsOutput as EmbeddingsOutput, GenerationInput as GenerationInput, GenerationOutput as GenerationOutput, HfAgentInput as HfAgentInput, MetadataOutput as MetadataOutput, unmarshal_vllm_outputs as unmarshal_vllm_outputs
from ._strategies import AmdGpuResource as AmdGpuResource, CascadingResourceStrategy as CascadingResourceStrategy, LiteralResourceSpec as LiteralResourceSpec, NvidiaGpuResource as NvidiaGpuResource, available_resource_spec as available_resource_spec, get_resource as get_resource
from .config import (
    CONFIG_MAPPING as CONFIG_MAPPING,
    CONFIG_MAPPING_NAMES as CONFIG_MAPPING_NAMES,
    START_BAICHUAN_COMMAND_DOCSTRING as START_BAICHUAN_COMMAND_DOCSTRING,
    START_CHATGLM_COMMAND_DOCSTRING as START_CHATGLM_COMMAND_DOCSTRING,
    START_DOLLY_V2_COMMAND_DOCSTRING as START_DOLLY_V2_COMMAND_DOCSTRING,
    START_FALCON_COMMAND_DOCSTRING as START_FALCON_COMMAND_DOCSTRING,
    START_FLAN_T5_COMMAND_DOCSTRING as START_FLAN_T5_COMMAND_DOCSTRING,
    START_GPT_NEOX_COMMAND_DOCSTRING as START_GPT_NEOX_COMMAND_DOCSTRING,
    START_LLAMA_COMMAND_DOCSTRING as START_LLAMA_COMMAND_DOCSTRING,
    START_MPT_COMMAND_DOCSTRING as START_MPT_COMMAND_DOCSTRING,
    START_OPT_COMMAND_DOCSTRING as START_OPT_COMMAND_DOCSTRING,
    START_STABLELM_COMMAND_DOCSTRING as START_STABLELM_COMMAND_DOCSTRING,
    START_STARCODER_COMMAND_DOCSTRING as START_STARCODER_COMMAND_DOCSTRING,
    AutoConfig as AutoConfig,
    BaichuanConfig as BaichuanConfig,
    ChatGLMConfig as ChatGLMConfig,
    DollyV2Config as DollyV2Config,
    FalconConfig as FalconConfig,
    FlanT5Config as FlanT5Config,
    GPTNeoXConfig as GPTNeoXConfig,
    LlamaConfig as LlamaConfig,
    MPTConfig as MPTConfig,
    OPTConfig as OPTConfig,
    StableLMConfig as StableLMConfig,
    StarCoderConfig as StarCoderConfig,
)
