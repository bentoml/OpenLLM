from typing import (
  Any,
  AsyncGenerator,
  Dict,
  Generic,
  Iterable,
  List,
  Literal,
  Optional,
  Protocol,
  Tuple,
  Type,
  TypeVar,
  Union,
  final,
)

from bentoml import Model, Strategy, Tag
from bentoml._internal.runner.runner_handle import RunnerHandle
from openllm_core import LLMConfig
from openllm_core._typing_compat import LiteralBackend, T, overload

from ._llm import LLM

try:
  from vllm import AsyncLLMEngine
except ImportError:
  AsyncLLMEngine = Any

try:
  from transformers import PreTrainedModel
except ImportError:
  PreTrainedModel = Any

Mo = TypeVar('Mo')

class _Runnable(Protocol[Mo]):
  SUPPORTED_RESOURCES: Tuple[Literal['nvidia.com/gpu'], Literal['amd.com/gpu'], Literal['cpu']] = ...
  SUPPORTS_CPU_MULTI_THREADING: bool = ...
  config: LLMConfig = ...
  model: Mo = ...
  def __init__(self, llm: LLM[Mo, T]) -> None: ...
  async def generate_iterator(
    self,
    prompt_token_ids: List[int],
    request_id: str,
    stop: Optional[Union[str, Iterable[str]]] = ...,
    adapter_name: Optional[str] = ...,
    **attrs: Any,
  ) -> AsyncGenerator[str, None]: ...

In = TypeVar('In')
Ret = TypeVar('Ret')

class RunnerMethod(Generic[In, Ret]): ...

@final
class vLLMRunnable(_Runnable[AsyncLLMEngine]): ...

@final
class PyTorchRunnable(_Runnable[PreTrainedModel]):
  tokenizer: Any

@overload
def runnable(backend: Literal['vllm']) -> Type[vLLMRunnable]: ...
@overload
def runnable(backend: Literal['pt']) -> Type[PyTorchRunnable]: ...
@overload
def runnable(backend: Optional[str] = ...) -> Type[Union[vLLMRunnable, PyTorchRunnable]]: ...

class Runner(Protocol[Mo, T]):
  __doc__: str = ...
  __module__: str = ...
  llm_type: str = ...
  llm_tag: Tag = ...
  identifying_params: Dict[str, Any] = ...
  llm: LLM[Mo, T] = ...
  config: LLMConfig = ...
  backend: LiteralBackend = ...
  has_adapters: bool = ...
  prompt_template: Optional[str] = ...
  system_message: Optional[str] = ...

  class generate_iterator(RunnerMethod[List[int], AsyncGenerator[str, None]]):
    @staticmethod
    def async_stream(
      prompt_token_ids: List[int],
      request_id: str,
      stop: Optional[Union[Iterable[str], str]] = ...,
      adapter_name: Optional[str] = ...,
      **attrs: Any,
    ) -> AsyncGenerator[str, None]: ...

  def __init__(
    self,
    runnable_class: Type[_Runnable[Mo]],
    *,
    runnable_init_params: Optional[Dict[str, Any]] = ...,
    name: Optional[str] = ...,
    scheduling_strategy: Type[Strategy] = ...,
    models: Optional[List[Model]] = ...,
    max_batch_size: Optional[int] = ...,
    max_latency_ms: Optional[int] = ...,
    method_configs: Optional[Dict[str, Dict[str, int]]] = ...,
    embedded: bool = ...,
  ) -> None: ...

  name: str = ...
  models: List[Model] = ...
  resource_config: Dict[str, Any]
  runnable_class: Type[_Runnable[Mo]]
  embedded: bool
  runner_methods: List[RunnerMethod[Any, Any]]
  scheduling_strategy: Type[Strategy]
  workers_per_resource: Union[int, float] = ...
  runnable_init_params: Dict[str, Any] = ...
  _runner_handle: RunnerHandle = ...

  def init_local(self, quiet: bool = False) -> None: ...
  def init_client(self, handle_class: Optional[Type[RunnerHandle]] = ..., *args: Any, **kwargs: Any) -> None: ...
  async def runner_handle_is_ready(self, timeout: int = ...) -> bool: ...
  def destroy(self) -> None: ...
  @property
  def scheduled_worker_count(self) -> int: ...
  @property
  def scheduled_worker_env_map(self) -> Dict[int, Dict[str, Any]]: ...
