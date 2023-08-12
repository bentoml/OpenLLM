from __future__ import annotations
import sys, typing as t, bentoml, attr, abc
from bentoml._internal.types import ModelSignatureDict as ModelSignatureDict
if t.TYPE_CHECKING:
  import openllm, peft, transformers, auto_gptq as autogptq, vllm
  from bentoml._internal.runner.runnable import RunnableMethod
  from bentoml._internal.runner.runner import RunnerMethod
  from bentoml._internal.runner.strategy import Strategy

  from .bundle.oci import LiteralContainerVersionStrategy
  from .utils.lazy import VersionInfo

M = t.TypeVar("M", bound="t.Union[transformers.PreTrainedModel, transformers.Pipeline, transformers.TFPreTrainedModel, transformers.FlaxPreTrainedModel, vllm.LLMEngine, vllm.AsyncLLMEngine, peft.PeftModel, autogptq.modeling.BaseGPTQForCausalLM]")
T = t.TypeVar("T", bound="t.Union[transformers.PreTrainedTokenizerFast, transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerBase]")

AnyCallable = t.Callable[..., t.Any]
DictStrAny = t.Dict[str, t.Any]
ListAny = t.List[t.Any]
ListStr = t.List[str]
TupleAny = t.Tuple[t.Any, ...]
At = t.TypeVar("At", bound=attr.AttrsInstance)

LiteralRuntime = t.Literal["pt", "tf", "flax", "vllm"]
AdapterType = t.Literal["lora", "adalora", "adaption_prompt", "prefix_tuning", "p_tuning", "prompt_tuning", "ia3"]

if sys.version_info[:2] >= (3,11):
  from typing import LiteralString as LiteralString, Self as Self, overload as overload
  from typing import NotRequired as NotRequired, Required as Required, dataclass_transform as dataclass_transform
else:
  from typing_extensions import LiteralString as LiteralString, Self as Self, overload as overload
  from typing_extensions import NotRequired as NotRequired, Required as Required, dataclass_transform as dataclass_transform

if sys.version_info[:2] >= (3,10):
  from typing import TypeAlias as TypeAlias, ParamSpec as ParamSpec, Concatenate as Concatenate
else:
  from typing_extensions import TypeAlias as TypeAlias, ParamSpec as ParamSpec, Concatenate as Concatenate

if sys.version_info[:2] >= (3,9):
  from typing import TypedDict as TypedDict
else:
  from typing_extensions import TypedDict as TypedDict

class PeftAdapterOutput(TypedDict):
  success: bool
  result: t.Dict[str, peft.PeftConfig]
  error_msg: str

class LLMEmbeddings(t.TypedDict):
  embeddings: t.List[t.List[float]]
  num_tokens: int

class AdaptersTuple(TupleAny):
  adapter_id: str
  name: t.Optional[str]
  config: DictStrAny

AdaptersMapping = t.Dict[AdapterType, t.Tuple[AdaptersTuple, ...]]

class RefTuple(TupleAny):
  git_hash: str
  version: VersionInfo
  strategy: LiteralContainerVersionStrategy

class LLMRunnable(bentoml.Runnable, t.Generic[M, T]):
  SUPPORTED_RESOURCES = ("amd.com/gpu", "nvidia.com/gpu", "cpu")
  SUPPORTS_CPU_MULTI_THREADING = True
  __call__: RunnableMethod[LLMRunnable[M, T], [str], list[t.Any]]
  set_adapter: RunnableMethod[LLMRunnable[M, T], [str], dict[t.Literal["success", "error_msg"], bool | str]]
  embeddings: RunnableMethod[LLMRunnable[M, T], [list[str]], LLMEmbeddings]
  generate: RunnableMethod[LLMRunnable[M, T], [str], list[t.Any]]
  generate_one: RunnableMethod[LLMRunnable[M, T], [str, list[str]], t.Sequence[dict[t.Literal["generated_text"], str]]]
  generate_iterator: RunnableMethod[LLMRunnable[M, T], [str], t.Generator[t.Any, None, None]]

class LLMRunner(bentoml.Runner, t.Generic[M, T]):
  __doc__: str
  __module__: str
  llm_type: str
  identifying_params: dict[str, t.Any]
  llm: openllm.LLM[M, T]
  config: openllm.LLMConfig
  implementation: LiteralRuntime
  supports_embeddings: bool
  supports_hf_agent: bool
  has_adapters: bool
  embeddings: RunnerMethod[LLMRunnable[M, T], [list[str]], LLMEmbeddings]
  generate: RunnerMethod[LLMRunnable[M, T], [str], list[t.Any]]
  generate_one: RunnerMethod[LLMRunnable[M, T], [str, list[str]], t.Sequence[dict[t.Literal["generated_text"], str]]]
  generate_iterator: RunnerMethod[LLMRunnable[M, T], [str], t.Generator[t.Any, None, None]]
  def __init__(self, runnable_class: type[LLMRunnable[M, T]], *, runnable_init_params: dict[str, t.Any] | None = ..., name: str | None = ..., scheduling_strategy: type[Strategy] = ..., models: list[bentoml.Model] | None = ..., max_batch_size: int | None = ..., max_latency_ms: int | None = ..., method_configs: dict[str, dict[str, int]] | None = ..., embedded: bool = False,) -> None: ...
  def __call__(self, prompt: str, **attrs: t.Any) -> t.Any: ...
  @abc.abstractmethod
  def embed(self, prompt: str | list[str]) -> LLMEmbeddings: ...
  def run(self, prompt: str, **attrs: t.Any) -> t.Any: ...
  async def async_run(self, prompt: str, **attrs: t.Any) -> t.Any: ...
  @abc.abstractmethod
  def download_model(self) -> bentoml.Model: ...
  @property
  @abc.abstractmethod
  def peft_adapters(self) -> PeftAdapterOutput: ...
  @property
  @abc.abstractmethod
  def __repr_keys__(self) -> set[str]: ...
