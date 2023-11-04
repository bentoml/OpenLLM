'''Schema definition for OpenLLM. This schema is used throughout openllm core components library.'''
from __future__ import annotations
import functools
import typing as t

import attr
import inflection
import orjson

from openllm_core._configuration import LLMConfig

from .utils import converter
from .utils import gen_random_uuid

if t.TYPE_CHECKING:
  import vllm

@attr.frozen(slots=True)
class GenerateInput:
  prompt: str
  llm_config: LLMConfig
  adapter_name: str | None = attr.field(default=None)

  def model_dump(self) -> dict[str, t.Any]:
    return {'prompt': self.prompt, 'llm_config': self.llm_config.model_dump(flatten=True), 'adapter_name': self.adapter_name}

  @staticmethod
  def convert_llm_config(data: dict[str, t.Any] | LLMConfig, cls: type[LLMConfig] | None = None) -> LLMConfig:
    if isinstance(data, LLMConfig): return data
    else:
      if cls is None: raise ValueError("'cls' must pass if given data is a dictionary.")
      return cls(**data)

  @classmethod
  def for_model(cls, model_name: str, **attrs: t.Any) -> type[GenerateInput]:
    import openllm
    return cls.from_llm_config(openllm.AutoConfig.for_model(model_name, **attrs))

  @classmethod
  def from_llm_config(cls, llm_config: LLMConfig) -> type[GenerateInput]:
    return attr.make_class(inflection.camelize(llm_config['model_name']) + 'GenerateInput',
                           attrs={
                               'prompt': attr.field(type=str),
                               'llm_config': attr.field(type=llm_config.__class__, default=llm_config, converter=functools.partial(cls.convert_llm_config, cls=llm_config.__class__)),
                               'adapter_name': attr.field(default=None, type=str)
                           })

@attr.frozen(slots=True)
class MetadataOutput:
  model_id: str
  timeout: int
  model_name: str
  backend: str
  configuration: str
  prompt_template: str
  system_message: str

@attr.define
class HfAgentInput:
  inputs: str
  parameters: t.Dict[str, t.Any]

FinishReason = t.Literal['length', 'stop']

# NOTE: vendor parameters from vllm.RequestOutput and vllm.CompletionOutput
# since vllm is not available on CPU.
# OpenLLM will adapt CPU outputs to similar architecture with vLLM outputs for consistency

SampleLogprobs = t.List[t.Dict[int, float]]
PromptLogprobs = t.List[t.Optional[t.Dict[int, float]]]

@attr.define
class CompletionChunk:
  index: int
  text: str
  token_ids: t.List[int]
  cumulative_logprob: float
  logprobs: t.Optional[SampleLogprobs] = None
  finish_reason: t.Optional[FinishReason] = None

@attr.define
class GenerationOutput:
  prompt: str
  finished: bool
  outputs: t.List[CompletionChunk]
  prompt_token_ids: t.Optional[t.List[int]] = attr.field(default=None)
  prompt_logprobs: t.Optional[PromptLogprobs] = attr.field(default=None)
  request_id: str = attr.field(factory=lambda: gen_random_uuid())

  @staticmethod
  def examples() -> GenerationOutput:
    return GenerationOutput(prompt='What is the meaning of life?',
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
  def from_sse(cls, sse_message: str) -> GenerationOutput:
    data = [line[6:] for line in sse_message.strip().split('\n') if line.startswith('data: ')]
    if not data: raise ValueError('No data found in SSE message.')
    if len(data) > 1: raise ValueError('Multiple data found in SSE message.')
    try:
      return converter.structure(orjson.loads(''.join(data)), cls)
    except orjson.JSONDecodeError as e:
      raise ValueError(f'Failed to parse JSON from SSE message: {sse_message}') from e

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
  def unmarshal(self)->dict[str, t.Any]:return converter.unstructure(self)
  def unmarshal_json(self)->str:return orjson.dumps(self.unmarshal()).decode('utf-8')
  # yapf: enable
