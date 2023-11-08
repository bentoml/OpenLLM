'''Schema definition for OpenLLM. This schema is used throughout openllm core components library.'''
from __future__ import annotations
import typing as t

import attr
import inflection
import orjson

from ._configuration import LLMConfig
from .config import AutoConfig
from .utils import converter
from .utils import gen_random_uuid

if t.TYPE_CHECKING:
  import vllm

  import openllm

  from ._typing_compat import M
  from ._typing_compat import T

@attr.frozen(slots=True)
class MetadataOutput:
  model_id: str
  timeout: int
  model_name: str
  backend: str
  configuration: str
  prompt_template: str
  system_message: str

  @classmethod
  def examples(cls, llm: openllm.LLM[M, T]) -> MetadataOutput:
    return cls(model_id=llm.model_id,
               timeout=llm.config['timeout'],
               model_name=llm.config['model_name'],
               backend=llm.__llm_backend__,
               configuration=llm.config.model_dump_json().decode(),
               prompt_template='{system_message}',
               system_message='You are a helpful assistant.')

  # yapf: disable
  def model_dump(self)->dict[str, t.Any]: return converter.unstructure(self)
  def model_dump_json(self)->str:return orjson.dumps(self.model_dump(),option=orjson.OPT_INDENT_2).decode('utf-8')
  # yapf: enable

@attr.define(slots=True, frozen=True)
class GenerationInput:
  prompt: str
  llm_config: LLMConfig
  stop: list[str] | None = attr.field(default=None)
  adapter_name: str | None = attr.field(default=None)

  # yapf: disable
  @classmethod
  def from_model(cls,model_name:str,**attrs: t.Any)->type[GenerationInput]:return cls.from_llm_config(AutoConfig.for_model(model_name,**attrs))
  def model_dump(self)->dict[str,t.Any]:return {'prompt': self.prompt,'stop': self.stop,'llm_config': self.llm_config.model_dump(flatten=True),'adapter_name': self.adapter_name}
  def model_dump_json(self)->str:return orjson.dumps(self.model_dump(),option=orjson.OPT_INDENT_2).decode('utf-8')
  # yapf: enable

  @classmethod
  def from_llm_config(cls, llm_config: LLMConfig) -> type[GenerationInput]:
    def init(self: GenerationInput, prompt: str, stop: list[str] | None, adapter_name: str | None) -> None:
      self.__attrs_init__(prompt=prompt, llm_config=llm_config, stop=stop, adapter_name=adapter_name)  # type: ignore

    def _llm_config_converter(data: dict[str, t.Any] | LLMConfig) -> LLMConfig:
      if isinstance(data, LLMConfig): return data
      return llm_config.__class__(**data)

    klass: type[GenerationInput] = attr.make_class(inflection.camelize(llm_config['model_name']) + 'GenerationInput', {
        '__init__': init,
        'llm_config': attr.field(default=llm_config, converter=_llm_config_converter)
    },
                                                   bases=(cls,),
                                                   slots=True,
                                                   weakref_slot=True,
                                                   frozen=True,
                                                   repr=True,
                                                   collect_by_mro=True)

    def examples(_: type[GenerationInput]) -> GenerationInput:
      return klass(prompt='What is the meaning of life?', llm_config=llm_config, stop=['\n'])

    setattr(klass, 'examples', classmethod(examples))

    try:
      klass.__module__ = cls.__module__
    except (AttributeError, ValueError):
      pass
    return klass

# NOTE: parameters from vllm.RequestOutput and vllm.CompletionOutput since vllm is not available on CPU.
# OpenLLM will adapt CPU outputs to similar architecture with vLLM outputs for consistency

SampleLogprobs = t.List[t.Dict[int, float]]
PromptLogprobs = t.List[t.Optional[t.Dict[int, float]]]
FinishReason = t.Literal['length', 'stop']

@attr.define
class CompletionChunk:
  index: int
  text: str
  token_ids: t.List[int]
  cumulative_logprob: float
  logprobs: t.Optional[SampleLogprobs] = None
  finish_reason: t.Optional[FinishReason] = None

  # yapf: disable
  def with_options(self,**options: t.Any)->CompletionChunk: return attr.evolve(self, **options)
  def model_dump(self)->dict[str, t.Any]:return converter.unstructure(self)
  def model_dump_json(self)->str:return orjson.dumps(self.model_dump(),option=orjson.OPT_NON_STR_KEYS).decode('utf-8')
  # yapf: enable

@attr.define
class GenerationOutput:
  prompt: str
  finished: bool
  outputs: t.List[CompletionChunk]
  prompt_token_ids: t.Optional[t.List[int]] = attr.field(default=None)
  prompt_logprobs: t.Optional[PromptLogprobs] = attr.field(default=None)
  request_id: str = attr.field(factory=lambda: gen_random_uuid())

  @classmethod
  def examples(cls) -> GenerationOutput:
    return cls(prompt='What is the meaning of life?',
               finished=True,
               outputs=[
                   CompletionChunk(index=0,
                                   text='\nLife is the process by which organisms, such as bacteria and cells, reproduce themselves and continue to exist.',
                                   token_ids=[50118, 12116, 16, 5, 609, 30, 61, 28340, 6, 215, 25, 9436, 8, 4590, 6, 33942, 1235, 8, 535],
                                   cumulative_logprob=0.0,
                                   logprobs=None,
                                   finish_reason='length')
               ],
               prompt_token_ids=[2, 2264, 16, 5, 3099, 9, 301, 116],
               prompt_logprobs=None,
               request_id=gen_random_uuid())

  @classmethod
  def from_runner(cls, data: str) -> GenerationOutput:
    if not data: raise ValueError('No data found from messages.')
    try:
      return converter.structure(orjson.loads(data), cls)
    except orjson.JSONDecodeError as e:
      raise ValueError(f'Failed to parse JSON from SSE message: {sse_message!r}') from e

  @classmethod
  def from_vllm(cls, request_output: vllm.RequestOutput) -> GenerationOutput:
    return cls(prompt=request_output.prompt,
               finished=request_output.finished,
               request_id=request_output.request_id,
               outputs=[
                   CompletionChunk(index=it.index, text=it.text, token_ids=it.token_ids, cumulative_logprob=it.cumulative_logprob, logprobs=it.logprobs, finish_reason=it.finish_reason)
                   for it in request_output.outputs
               ],
               prompt_token_ids=request_output.prompt_token_ids,
               prompt_logprobs=request_output.prompt_logprobs)

  # yapf: disable
  def with_options(self,**options: t.Any)->GenerationOutput: return attr.evolve(self, **options)
  def model_dump(self)->dict[str, t.Any]:return converter.unstructure(self)
  def model_dump_json(self)->str:return orjson.dumps(self.model_dump(),option=orjson.OPT_NON_STR_KEYS).decode('utf-8')
  # yapf: enable
