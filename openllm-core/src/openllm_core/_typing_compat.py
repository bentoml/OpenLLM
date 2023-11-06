# mypy: disable-error-code="type-arg,valid-type"
from __future__ import annotations
import abc
import sys
import typing as t

import attr

_is_bentoml_installed = False
try:
  import bentoml
  _is_bentoml_installed = True
except ImportError:
  bentoml = None
  _is_bentoml_installed = False

if t.TYPE_CHECKING:
  import peft
  import transformers

  import openllm

  from bentoml._internal.runner.runnable import RunnableMethod
  from bentoml._internal.runner.runner import RunnerMethod
  from bentoml._internal.runner.strategy import Strategy

  from .utils.lazy import VersionInfo

if t.TYPE_CHECKING:
  from types import UnionType

  from bentoml._internal.types import LazyType

  AnyType: t.TypeAlias = t.Type[t.Any] | UnionType | LazyType[t.Any]
else:
  AnyType = t.Any

M = t.TypeVar('M', bound='t.Union[transformers.PreTrainedModel, peft.PeftModel]')
T = t.TypeVar('T', bound='t.Union[transformers.PreTrainedTokenizerFast, transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerBase]')

def get_literal_args(typ: t.Any) -> tuple[str, ...]:
  return getattr(typ, '__args__')

AnyCallable = t.Callable[..., t.Any]
DictStrAny = t.Dict[str, t.Any]
ListAny = t.List[t.Any]
ListStr = t.List[str]
TupleAny = t.Tuple[t.Any, ...]
At = t.TypeVar('At', bound=attr.AttrsInstance)

LiteralSerialisation = t.Literal['safetensors', 'legacy']
LiteralQuantise = t.Literal['int8', 'int4', 'gptq', 'awq']
LiteralBackend = t.Literal['pt', 'vllm', 'ggml', 'mlc']
AdapterType = t.Literal['lora', 'adalora', 'adaption_prompt', 'prefix_tuning', 'p_tuning', 'prompt_tuning', 'ia3']

# TODO: support quay
LiteralContainerRegistry = t.Literal['docker', 'gh', 'ecr']
LiteralContainerVersionStrategy = t.Literal['release', 'nightly', 'latest', 'custom']

LiteralResourceSpec = t.Literal['cloud-tpus.google.com/v2', 'amd.com/gpu', 'nvidia.com/gpu', 'cpu']

InferenceReturnType = t.Literal['text', 'object', 'sse', 'token']

if sys.version_info[:2] >= (3, 11):
  from typing import LiteralString as LiteralString
  from typing import NotRequired as NotRequired
  from typing import Required as Required
  from typing import Self as Self
  from typing import dataclass_transform as dataclass_transform
  from typing import overload as overload
else:
  from typing_extensions import LiteralString as LiteralString
  from typing_extensions import NotRequired as NotRequired
  from typing_extensions import Required as Required
  from typing_extensions import Self as Self
  from typing_extensions import dataclass_transform as dataclass_transform
  from typing_extensions import overload as overload

if sys.version_info[:2] >= (3, 10):
  from typing import Concatenate as Concatenate
  from typing import ParamSpec as ParamSpec
  from typing import TypeAlias as TypeAlias
else:
  from typing_extensions import Concatenate as Concatenate
  from typing_extensions import ParamSpec as ParamSpec
  from typing_extensions import TypeAlias as TypeAlias

class ModelSignatureDict(t.TypedDict, total=False):
  batchable: bool
  batch_dim: t.Union[t.Tuple[int, int], int]
  input_spec: t.Optional[t.Union[t.Tuple[AnyType], AnyType]]
  output_spec: t.Optional[AnyType]

class PeftAdapterOutput(t.TypedDict):
  success: bool
  result: t.Dict[str, peft.PeftConfig]
  error_msg: str

class AdaptersTuple(TupleAny):
  adapter_id: str
  name: t.Optional[str]
  config: DictStrAny

AdaptersMapping = t.Dict[AdapterType, t.Tuple[AdaptersTuple, ...]]

class RefTuple(TupleAny):
  git_hash: str
  version: VersionInfo
  strategy: LiteralContainerVersionStrategy

if _is_bentoml_installed:

  class LLMRunnable(bentoml.Runnable, t.Generic[M, T]):
    SUPPORTED_RESOURCES = ('amd.com/gpu', 'nvidia.com/gpu', 'cpu')
    SUPPORTS_CPU_MULTI_THREADING = True
    generate: RunnableMethod[LLMRunnable[M, T], [list[int], str, str | t.Iterable[str] | None], t.AsyncGenerator[str, None]]
    generate_iterator: RunnableMethod[LLMRunnable[M, T], [list[int], str, str | t.Iterable[str] | None], t.AsyncGenerator[str, None]]
    # generate_one: RunnableMethod[LLMRunnable[M, T], [str, list[str]], t.Sequence[dict[t.Literal['generated_text'], str]]]

  class LLMRunner(bentoml.Runner, t.Generic[M, T]):
    __doc__: str
    __module__: str
    llm_type: str
    llm_tag: bentoml.Tag
    identifying_params: dict[str, t.Any]
    llm: openllm.LLM[M, T]
    config: openllm.LLMConfig
    backend: LiteralBackend
    has_adapters: bool
    system_message: str | None
    prompt_template: str | None
    generate: RunnerMethod[LLMRunnable[M, T], [list[int], str, str | t.Iterable[str] | None], t.AsyncGenerator[str, None]]
    generate_iterator: RunnerMethod[LLMRunnable[M, T], [list[int], str, str | t.Iterable[str] | None], t.AsyncGenerator[str, None]]

    # generate_one: RunnerMethod[LLMRunnable[M, T], [str, list[str]], t.Sequence[dict[t.Literal['generated_text'], str]]]

    def __init__(self,
                 runnable_class: type[LLMRunnable[M, T]],
                 *,
                 runnable_init_params: dict[str, t.Any] | None = ...,
                 name: str | None = ...,
                 scheduling_strategy: type[Strategy] = ...,
                 models: list[bentoml.Model] | None = ...,
                 max_batch_size: int | None = ...,
                 max_latency_ms: int | None = ...,
                 method_configs: dict[str, dict[str, int]] | None = ...,
                 embedded: bool = False,
                 ) -> None:
      ...

    @abc.abstractmethod
    def download_model(self) -> bentoml.Model:
      ...

    @property
    @abc.abstractmethod
    def peft_adapters(self) -> PeftAdapterOutput:
      ...

    @property
    @abc.abstractmethod
    def __repr_keys__(self) -> set[str]:
      ...
else:
  # NOTE: t.Any is also a type
  LLMRunnable = LLMRunner = t.Any
