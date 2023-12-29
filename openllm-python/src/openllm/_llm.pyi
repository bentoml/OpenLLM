from typing import Any, AsyncGenerator, Dict, Generic, Iterable, List, Literal, Optional, Tuple, TypedDict, Union
from _bentoml_sdk.service import ServiceConfig
from _bentoml_sdk.service.dependency import Dependency
from _bentoml_sdk.service.factory import Service

import attr
import torch
from peft.config import PeftConfig
from peft.peft_model import PeftModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM

from bentoml import Model, Tag
from openllm_core import LLMConfig
from openllm_core._schemas import GenerationOutput
from openllm_core._typing_compat import AdapterMap, AdapterType, LiteralBackend, LiteralDtype, LiteralQuantise, LiteralSerialisation, M, T

from ._quantisation import QuantizationConfig
from ._runners import Runner

InjectedModel = Union[PeftModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM]

class IdentifyingParams(TypedDict):
  configuration: str
  model_ids: str
  model_id: str

ResolvedAdapterMap = Dict[AdapterType, Dict[str, Tuple[PeftConfig, str]]]
CTranslateDtype = Literal['int8_float32', 'int8_float16', 'int8_bfloat16']
Dtype = Union[LiteralDtype, CTranslateDtype, Literal['auto', 'half', 'float']]

@attr.define(slots=True, repr=False, init=False)
class LLM(Generic[M, T]):
  _model_id: str
  _revision: Optional[str]
  _quantization_config: Optional[QuantizationConfig]
  _quantise: Optional[LiteralQuantise]
  _model_decls: Tuple[Any, ...]
  __model_attrs: Dict[str, Any]
  __tokenizer_attrs: Dict[str, Any]
  _tag: Tag
  _adapter_map: Optional[AdapterMap]
  _serialisation: LiteralSerialisation
  _local: bool
  _max_model_len: Optional[int]
  _gpu_memory_utilization: float

  __llm_dtype__: Dtype = ...
  __llm_torch_dtype__: Optional[torch.dtype] = ...
  __llm_config__: Optional[LLMConfig] = ...
  __llm_backend__: LiteralBackend = ...
  __llm_quantization_config__: Optional[QuantizationConfig] = ...
  __llm_runner__: Optional[Runner[M, T]] = ...
  __llm_model__: Optional[M] = ...
  __llm_tokenizer__: Optional[T] = ...
  __llm_adapter_map__: Optional[ResolvedAdapterMap] = ...
  __llm_trust_remote_code__: bool = ...
  __llm_services_config__: Optional[Dict[str, Any]] = ...

  def __repr__(self) -> str: ...
  def __init__(
    self,
    model_id: str,
    model_version: Optional[str] = ...,
    model_tag: Optional[Union[str, Tag]] = ...,
    llm_config: Optional[LLMConfig] = ...,
    backend: Optional[LiteralBackend] = ...,
    *args: Any,
    quantize: Optional[LiteralQuantise] = ...,
    quantization_config: Optional[QuantizationConfig] = ...,
    adapter_map: Optional[Dict[str, str]] = ...,
    serialisation: LiteralSerialisation = ...,
    trust_remote_code: bool = ...,
    embedded: bool = ...,
    dtype: Dtype = ...,
    low_cpu_mem_usage: bool = ...,
    max_model_len: Optional[int] = ...,
    gpu_memory_utilization: float = ...,
    services_config: Optional[Dict[str, Any]] = ...,
    **attrs: Any,
  ) -> None: ...
  @property
  def _torch_dtype(self) -> torch.dtype: ...
  @property
  def _model_attrs(self) -> Dict[str, Any]: ...
  @_model_attrs.setter
  def _model_attrs(self, model_attrs: Dict[str, Any]) -> None: ...
  @property
  def _tokenizer_attrs(self) -> Dict[str, Any]: ...
  @property
  def import_kwargs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]: ...
  @property
  def trust_remote_code(self) -> bool: ...
  @property
  def model_id(self) -> str: ...
  @property
  def revision(self) -> str: ...
  @property
  def tag(self) -> Tag: ...
  @property
  def bentomodel(self) -> Model: ...
  @property
  def quantization_config(self) -> QuantizationConfig: ...
  @property
  def has_adapters(self) -> bool: ...
  @property
  def local(self) -> bool: ...
  @property
  def quantise(self) -> Optional[LiteralQuantise]: ...
  @property
  def llm_type(self) -> str: ...
  @property
  def identifying_params(self) -> IdentifyingParams: ...
  @property
  def llm_parameters(self) -> Tuple[Tuple[Tuple[Any, ...], Dict[str, Any]], Dict[str, Any]]: ...
  @property
  def config(self) -> LLMConfig: ...
  @property
  def tokenizer(self) -> T: ...
  @property
  def model(self) -> M: ...
  @property
  def runner(self) -> Dependency[Runner[M, T]]: ...
  @property
  def service(self) -> Service[Any]: ...
  @property
  def adapter_map(self) -> ResolvedAdapterMap: ...
  def prepare(self, adapter_type: AdapterType = ..., use_gradient_checking: bool = ..., **attrs: Any) -> Tuple[InjectedModel, T]: ...
  async def generate(
    self,
    prompt: Optional[str],
    prompt_token_ids: Optional[List[int]] = ...,
    stop: Optional[Union[str, Iterable[str]]] = ...,
    stop_token_ids: Optional[List[int]] = ...,
    request_id: Optional[str] = ...,
    adapter_name: Optional[str] = ...,
    **attrs: Any,
  ) -> GenerationOutput: ...
  async def generate_iterator(
    self,
    prompt: Optional[str],
    prompt_token_ids: Optional[List[int]] = ...,
    stop: Optional[Union[str, Iterable[str]]] = ...,
    stop_token_ids: Optional[List[int]] = ...,
    request_id: Optional[str] = ...,
    adapter_name: Optional[str] = ...,
    **attrs: Any,
  ) -> AsyncGenerator[GenerationOutput, None]: ...
  class services_config:
    @staticmethod
    def runner(self: LLM[M, T]) -> ServiceConfig: ...
    @staticmethod
    def service(self: LLM[M, T]) -> ServiceConfig: ...
