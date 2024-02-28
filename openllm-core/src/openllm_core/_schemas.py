from __future__ import annotations

import pydantic, inflection, orjson, typing as t
from ._configuration import LLMConfig
from .utils import gen_random_uuid
from ._typing_compat import Required, TypedDict, LiteralString

if t.TYPE_CHECKING:
  import vllm


class MessageParam(TypedDict):
  role: t.Union[t.Literal['system', 'user', 'assistant'], LiteralString]
  content: str


class MessagesConverterInput(TypedDict):
  add_generation_prompt: bool
  messages: t.List[MessageParam]


class MetadataOutput(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(protected_namespaces=())
  model_id: str
  timeout: int
  model_name: str
  backend: str
  configuration: str


class GenerationInputDict(TypedDict, total=False):
  prompt: t.Optional[str]
  prompt_token_ids: t.Optional[t.List[int]]
  llm_config: Required[t.Dict[str, t.Any]]
  stop: t.Optional[t.List[str]]
  stop_token_ids: t.Optional[t.List[int]]
  request_id: t.Optional[str]
  adapter_name: t.Optional[str]


class GenerationInput(pydantic.BaseModel):
  prompt: t.Optional[str] = pydantic.Field(default=None)
  llm_config: LLMConfig = pydantic.Field(default_factory=dict)
  prompt_token_ids: t.Optional[t.List[int]] = pydantic.Field(default=None)
  stop: t.Optional[t.List[str]] = pydantic.Field(default=None)
  stop_token_ids: t.Optional[t.List[int]] = pydantic.Field(default=None)
  request_id: t.Optional[str] = pydantic.Field(default=None)
  adapter_name: t.Optional[str] = pydantic.Field(default=None)

  _class_ref: t.ClassVar[type[LLMConfig]] = pydantic.PrivateAttr()

  @pydantic.field_validator('llm_config')
  @classmethod
  def llm_config_validator(cls, v: LLMConfig | dict[str, t.Any]) -> LLMConfig:
    if isinstance(v, dict):
      return cls._class_ref.model_construct_env(**v)
    return v

  @pydantic.field_validator('stop')
  @classmethod
  def stop_validator(cls, data: str | list[str] | t.Iterable[str] | None) -> list[str] | None:
    if data is None:
      return None
    if isinstance(data, str):
      return [data]
    else:
      return list(data)

  @pydantic.model_serializer
  def ser_model(self) -> dict[str, t.Any]:
    flattened = self.llm_config.model_dump()
    flattened.update({
      'prompt': self.prompt,
      'prompt_token_ids': self.prompt_token_ids,
      'request_id': self.request_id,
      'adapter_name': self.adapter_name,
    })
    if self.stop is not None:
      flattened['stop'] = self.stop
    if self.stop_token_ids is not None:
      flattened['stop_token_ids'] = self.stop_token_ids
    return flattened

  def __init__(self, /, *, _internal: bool = False, **data: t.Any) -> None:
    if not _internal:
      raise RuntimeError(
        f'Cannot instantiate GenerationInput directly. Use "{self.__class__.__qualname__}.from_dict" instead.'
      )
    super().__init__(**data)

  @classmethod
  def from_dict(cls, structured: GenerationInputDict) -> GenerationInput:
    if not hasattr(cls, '_class_ref'):
      raise ValueError(
        'Cannot use "from_dict" from a raw GenerationInput class. Currently only supports class created from "from_config".'
      )
    filtered: dict[str, t.Any] = {k: v for k, v in structured.items() if v is not None}
    llm_config: dict[str, t.Any] | None = filtered.pop('llm_config', None)
    if llm_config is not None:
      filtered['llm_config'] = cls._class_ref.model_construct_env(**llm_config)

    return cls(_internal=True, **filtered)

  @classmethod
  def from_config(cls, llm_config: LLMConfig) -> type[GenerationInput]:
    klass = pydantic.create_model(
      inflection.camelize(llm_config['start_name']) + 'GenerationInput',
      __base__=cls,
      llm_config=(type(llm_config), llm_config),
      _class_ref=(llm_config.__class__, pydantic.PrivateAttr(default=llm_config.__class__)),
    )
    return klass


# NOTE: parameters from vllm.RequestOutput and vllm.CompletionOutput since vllm is not available on CPU.
# OpenLLM will adapt CPU outputs to similar architecture with vLLM outputs for consistency

SampleLogprobs = t.List[t.Dict[int, float]]
PromptLogprobs = t.List[t.Optional[t.Dict[int, float]]]
FinishReason = t.Literal['length', 'stop']


class CompletionChunk(pydantic.BaseModel):
  index: int
  text: str
  token_ids: t.List[int]
  cumulative_logprob: float
  logprobs: t.Optional[SampleLogprobs] = pydantic.Field(default=None)
  finish_reason: t.Optional[FinishReason] = pydantic.Field(default=None)


class GenerationOutput(pydantic.BaseModel):
  prompt: str
  finished: bool
  outputs: t.List[CompletionChunk]
  prompt_token_ids: t.Optional[t.List[int]] = pydantic.Field(default=None)
  prompt_logprobs: t.Optional[PromptLogprobs] = pydantic.Field(default=None)
  request_id: str = pydantic.Field(default_factory=gen_random_uuid)

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
      structured['prompt_logprobs'] = [
        {int(k): v for k, v in it.items()} if it else None for it in structured['prompt_logprobs']
      ]
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
