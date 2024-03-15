from __future__ import annotations
import typing as t

import attr
import inflection
import orjson

from ._configuration import LLMConfig
from .config import AutoConfig
from .utils import converter, gen_random_uuid

if t.TYPE_CHECKING:
  import vllm

  from ._typing_compat import Self, LiteralString


class MessageParam(t.TypedDict):
  role: t.Union[t.Literal['system', 'user', 'assistant'], LiteralString]
  content: str


@attr.define
class _SchemaMixin:
  def model_dump(self) -> dict[str, t.Any]:
    return converter.unstructure(self)

  def model_dump_json(self) -> str:
    return orjson.dumps(self.model_dump(), option=orjson.OPT_INDENT_2).decode('utf-8')

  def with_options(self, **options: t.Any) -> Self:
    return attr.evolve(self, **options)


@attr.define
class MetadataOutput(_SchemaMixin):
  model_id: str
  timeout: int
  model_name: str
  backend: str
  configuration: str

  def model_dump(self) -> dict[str, t.Any]:
    return {
      'model_id': self.model_id,
      'timeout': self.timeout,
      'model_name': self.model_name,
      'backend': self.backend,
      'configuration': self.configuration,
    }


@attr.define
class GenerationInput(_SchemaMixin):
  prompt: str
  llm_config: LLMConfig
  stop: list[str] | None = attr.field(default=None)
  adapter_name: str | None = attr.field(default=None)

  @classmethod
  def from_model(cls, model_name: str, **attrs: t.Any) -> type[GenerationInput]:
    return cls.from_llm_config(AutoConfig.for_model(model_name, **attrs))

  def model_dump(self) -> dict[str, t.Any]:
    return {'prompt': self.prompt, 'stop': self.stop, 'llm_config': self.llm_config.model_dump(flatten=True), 'adapter_name': self.adapter_name}

  @classmethod
  def from_llm_config(cls, llm_config: LLMConfig) -> type[GenerationInput]:
    def init(self: GenerationInput, prompt: str, stop: list[str] | None, adapter_name: str | None) -> None:
      self.__attrs_init__(prompt=prompt, llm_config=llm_config, stop=stop, adapter_name=adapter_name)  # type: ignore

    def _llm_config_converter(data: dict[str, t.Any] | LLMConfig) -> LLMConfig:
      if isinstance(data, LLMConfig):
        return data
      return llm_config.__class__(**data)

    klass: type[GenerationInput] = attr.make_class(
      inflection.camelize(llm_config['model_name']) + 'GenerationInput',
      {'__init__': init, 'llm_config': attr.field(default=llm_config, converter=_llm_config_converter)},
      bases=(cls,),
      slots=True,
      weakref_slot=True,
      frozen=True,
      repr=True,
      collect_by_mro=True,
    )

    def examples(_: type[GenerationInput]) -> dict[str, t.Any]:
      return klass(prompt='What is the meaning of life?', llm_config=llm_config, stop=[]).model_dump()

    klass.examples = classmethod(examples)

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
class CompletionChunk(_SchemaMixin):
  index: int
  text: str
  token_ids: t.List[int]
  cumulative_logprob: float
  logprobs: t.Optional[SampleLogprobs] = None
  finish_reason: t.Optional[FinishReason] = None

  def model_dump_json(self) -> str:
    return orjson.dumps(self.model_dump(), option=orjson.OPT_NON_STR_KEYS).decode('utf-8')


@attr.define
class GenerationOutput(_SchemaMixin):
  prompt: str
  finished: bool
  outputs: t.List[CompletionChunk]
  prompt_token_ids: t.Optional[t.List[int]] = attr.field(default=None)
  prompt_logprobs: t.Optional[PromptLogprobs] = attr.field(default=None)
  request_id: str = attr.field(factory=lambda: gen_random_uuid())

  @classmethod
  def examples(cls) -> dict[str, t.Any]:
    return cls(
      prompt='What is the meaning of life?',
      finished=True,
      outputs=[
        CompletionChunk(
          index=0,
          text='\nLife is the process by which organisms, such as bacteria and cells, reproduce themselves and continue to exist.',
          token_ids=[50118, 12116, 16, 5, 609, 30, 61, 28340, 6, 215, 25, 9436, 8, 4590, 6, 33942, 1235, 8, 535],
          cumulative_logprob=0.0,
          logprobs=None,
          finish_reason='length',
        )
      ],
      prompt_token_ids=[2, 2264, 16, 5, 3099, 9, 301, 116],
      prompt_logprobs=None,
      request_id=gen_random_uuid(),
    ).model_dump()

  @staticmethod
  def _preprocess_sse_message(data: str) -> str:
    proc = [line[6:] for line in data.strip().split('\n') if line.startswith('data: ')]
    if not proc:
      return data
    if len(proc) > 1:
      raise ValueError('Multiple data found in SSE message.')
    return proc[0]

  @classmethod
  def from_runner(cls, data: str) -> GenerationOutput:
    data = cls._preprocess_sse_message(data)
    if not data:
      raise ValueError('No data found from messages.')
    try:
      structured = orjson.loads(data)
    except orjson.JSONDecodeError as e:
      raise ValueError(f'Failed to parse JSON from SSE message: {data!r}') from e

    return cls.from_dict(structured)

  @classmethod
  def from_dict(cls, structured: dict[str, t.Any]) -> GenerationOutput:
    if structured['prompt_logprobs']:
      structured['prompt_logprobs'] = [{int(k): v for k, v in it.items()} if it else None for it in structured['prompt_logprobs']]
    return cls(
      prompt=structured['prompt'],
      finished=structured['finished'],
      prompt_token_ids=structured['prompt_token_ids'],
      prompt_logprobs=structured['prompt_logprobs'],
      request_id=structured['request_id'],
      outputs=[
        CompletionChunk(
          index=it['index'],
          text=it['text'],
          token_ids=it['token_ids'],
          cumulative_logprob=it['cumulative_logprob'],
          finish_reason=it['finish_reason'],
          logprobs=[{int(k): v for k, v in s.items()} for s in it['logprobs']] if it['logprobs'] else None,
        )
        for it in structured['outputs']
      ],
    )

  @classmethod
  def from_vllm(cls, request_output: vllm.RequestOutput) -> GenerationOutput:
    return cls(
      prompt=request_output.prompt,
      finished=request_output.finished,
      request_id=request_output.request_id,
      prompt_token_ids=request_output.prompt_token_ids,
      prompt_logprobs=request_output.prompt_logprobs,
      outputs=[
        CompletionChunk(
          index=it.index,
          text=it.text,
          token_ids=it.token_ids,
          cumulative_logprob=it.cumulative_logprob,
          logprobs=it.logprobs,
          finish_reason=it.finish_reason,
        )
        for it in request_output.outputs
      ],
    )

  def model_dump_json(self) -> str:
    return orjson.dumps(self.model_dump(), option=orjson.OPT_NON_STR_KEYS).decode('utf-8')


converter.register_structure_hook_func(lambda cls: attr.has(cls) and issubclass(cls, GenerationOutput), lambda data, cls: cls.from_dict(data))
