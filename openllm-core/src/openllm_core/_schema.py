'''Schema definition for OpenLLM. This schema is used throughout openllm core components library.'''
from __future__ import annotations
import functools
import typing as t

import attr
import inflection

from openllm_core._configuration import GenerationConfig
from openllm_core._configuration import LLMConfig

from .utils import bentoml_cattr
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
class GenerateOutput:
  responses: t.List[t.Any]
  configuration: t.Dict[str, t.Any]

  @property
  def marshaled_config(self) -> GenerationConfig:
    return bentoml_cattr.structure(self.configuration, GenerationConfig)

  @property
  def unmarshaled(self) -> dict[str, t.Any]:
    return bentoml_cattr.unstructure(self)

  def __getitem__(self, key: str) -> t.Any:
    if hasattr(self, key): return getattr(self, key)
    elif key in self.configuration: return self.configuration[key]
    else: raise KeyError(key)

@attr.frozen(slots=True)
class MetadataOutput:
  model_id: str
  timeout: int
  model_name: str
  backend: str
  configuration: str
  supports_hf_agent: bool
  prompt_template: str
  system_message: str

def unmarshal_vllm_outputs(request_output: vllm.RequestOutput) -> dict[str, t.Any]:
  return dict(request_id=request_output.request_id,
              prompt=request_output.prompt,
              finished=request_output.finished,
              prompt_token_ids=request_output.prompt_token_ids,
              outputs=[
                  dict(index=it.index, text=it.text, token_ids=it.token_ids, cumulative_logprob=it.cumulative_logprob, logprobs=it.logprobs, finish_reason=it.finish_reason)
                  for it in request_output.outputs
              ])

@attr.define
class HfAgentInput:
  inputs: str
  parameters: t.Dict[str, t.Any]

FinishReason = t.Literal['length', 'stop']

@attr.define
class CompletionChunk:
  index: int
  text: str
  token_ids: t.List[int]
  cumulative_logprob: float
  logprobs: t.Optional[t.List[t.Dict[int, float]]] = None
  finish_reason: t.Optional[FinishReason] = None

@attr.define
class GenerationOutput:
  prompt: str
  finished: bool
  outputs: t.List[CompletionChunk]
  prompt_token_ids: t.Optional[t.List[int]] = attr.field(default=None)
  request_id: str = attr.field(factory=lambda: gen_random_uuid())
