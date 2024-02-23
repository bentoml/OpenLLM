from __future__ import annotations

import inspect, orjson, bentoml, pydantic, openllm_core, traceback, openllm, functools, typing as t
from openllm_core.utils import check_bool_env, normalise_model_name, LazyLoader, gen_random_uuid
from openllm_core._typing_compat import LiteralQuantise, LiteralSerialisation, LiteralDtype
from openllm_core._schemas import GenerationOutput
from _bentoml_sdk.service import ServiceConfig

if t.TYPE_CHECKING:
  import vllm
else:
  vllm = LazyLoader('vllm', globals(), 'vllm')

Dtype = t.Union[LiteralDtype, t.Literal['auto', 'half', 'float']]


class LLM(pydantic.BaseModel):
  model_id: str
  tag: bentoml.Tag
  revision: str
  local: bool
  serialisation: LiteralSerialisation
  config: openllm_core.LLMConfig
  dtype: Dtype
  quantise: t.Optional[LiteralQuantise] = pydantic.Field(default=None)
  max_model_len: t.Optional[int] = pydantic.Field(default=None)
  gpu_memory_utilization: float = pydantic.Field(default=0.9)
  trust_remote_code: bool = pydantic.Field(default=False)
  service_config: t.Optional[ServiceConfig] = pydantic.Field(default_factory=dict)

  def __init__(self, _internal: bool = False, **kwargs: t.Any) -> None:
    if not _internal:
      raise RuntimeError(
        'Cannot instantiate LLM directly in the new API. Make sure to use `openllm.LLM.from_model` instead.'
      )
    super().__init__(**kwargs)

  def model_post_init(self) -> None:
    self.bentomodel = bentoml.models.get(self.tag)
    path = self.bentomodel.path if self.local else self.model_id

    num_gpus, dev = 1, openllm.utils.device_count()
    if dev >= 2:
      num_gpus = min(dev // 2 * 2, dev)
    quantise = self.quantise if self.quantise and self.quantise in {'gptq', 'awq', 'squeezellm'} else None
    dtype = 'float16' if quantise == 'gptq' else self.dtype  # NOTE: quantise GPTQ doesn't support bfloat16 yet.
    try:
      self.model = vllm.AsyncLLMEngine.from_engine_args(
        vllm.AsyncEngineArgs(
          worker_use_ray=False,
          engine_use_ray=False,
          tokenizer_mode='auto',
          tensor_parallel_size=num_gpus,
          model=path,
          tokenizer=path,
          trust_remote_code=self.trust_remote_code,
          dtype=dtype,
          max_model_len=self.max_model_len,
          gpu_memory_utilization=self.gpu_memory_utilization,
          quantization=quantise,
        )
      )
    except Exception as err:
      traceback.print_exc()
      raise openllm.exceptions.OpenLLMException(
        f'Failed to initialise vLLMEngine due to the following error:\n{err}'
      ) from err

  @classmethod
  def from_model(
    cls,
    model: bentoml.Model,
    llm_config: openllm_core.LLMConfig | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.9,
    service_config: ServiceConfig | None = None,
  ) -> LLM:
    metadata = model.info.metadata
    dtype = metadata['dtype']
    local = metadata['_local']
    model_id = metadata['model_id']
    trust_remote_code = check_bool_env('TRUST_REMOTE_CODE', default=metadata['trust_remote_code'])

    if llm_config is None:
      llm_config = openllm_core.AutoConfig.from_bentomodel(model)
    return cls(
      _internal=True,
      model_id=model_id,
      tag=model.tag,
      local=local,
      quantise=metadata.get('quantize'),
      serialisation=metadata['serialisation'],
      config=llm_config,
      dtype=dtype,
      revision=metadata['_revision'],
      max_model_len=max_model_len,
      gpu_memory_utilization=gpu_memory_utilization,
      trust_remote_code=trust_remote_code,
      service_config=service_config,
    )

  @property
  def llm_type(self) -> str:
    return normalise_model_name(self.model_id)

  @property
  def identifying_params(self) -> dict[str, str]:
    return {
      'configuration': self.config.model_dump_json(),
      'model_ids': orjson.dumps(self.config['model_ids']).decode(),
      'model_id': self.model_id,
    }

  @functools.cached_property
  def tokenizer(self):
    return self.model.tokenizer

  async def generate_iterator(
    self,
    prompt: str | None,
    prompt_token_ids: list[int] | None = None,
    stop: str | t.Iterable[str] | None = None,
    stop_token_ids: list[int] | None = None,
    request_id: str | None = None,
    adapter_name: str | None = None,
    **attrs: t.Any,
  ) -> t.AsyncGenerator[GenerationOutput, None]:
    config = self.config.model_construct_env(**attrs)

    stop_token_ids = stop_token_ids or []
    eos_token_id = attrs.get('eos_token_id', config['eos_token_id'])
    if eos_token_id and not isinstance(eos_token_id, list):
      eos_token_id = [eos_token_id]
    stop_token_ids.extend(eos_token_id or [])
    if (config_eos := config['eos_token_id']) and config_eos not in stop_token_ids:
      stop_token_ids.append(config_eos)
    if self.tokenizer.eos_token_id not in stop_token_ids:
      stop_token_ids.append(self.tokenizer.eos_token_id)
    if stop is None:
      stop = set()
    elif isinstance(stop, str):
      stop = {stop}
    else:
      stop = set(stop)

    request_id = gen_random_uuid() if request_id is None else request_id
    previous_texts, previous_num_tokens = [''] * config['n'], [0] * config['n']

    config = config.model_construct_env(stop=list(stop), stop_token_ids=stop_token_ids)
    top_p = 1.0 if config['temperature'] <= 1e-5 else config['top_p']
    generation_config = config.generation_config.model_copy(update={'top_p': top_p})
    sampling_params = vllm.SamplingParams(**{
      k: getattr(generation_config, k) for k in set(inspect.signature(vllm.SamplingParams).parameters.keys())
    })

    try:
      generator = self.model.generate(
        prompt, sampling_params=sampling_params, request_id=request_id, prompt_token_ids=prompt_token_ids
      )
    except Exception as err:
      raise RuntimeError(f'Failed to start generation task: {err}') from err

    try:
      async for request_output in generator:
        generated = GenerationOutput.from_vllm(request_output).model_copy(update=dict(prompt=prompt))
        delta_outputs = [None] * len(generated.outputs)
        for output in generated.outputs:
          i = output.index
          delta_tokens, delta_text = output.token_ids[previous_num_tokens[i] :], output.text[len(previous_texts[i]) :]
          previous_texts[i], previous_num_tokens[i] = output.text, len(output.token_ids)
          delta_outputs[i] = output.model_copy(update=dict(text=delta_text, token_ids=delta_tokens))
        yield generated.model_copy(update=dict(outputs=delta_outputs))
    except Exception as err:
      raise RuntimeError(f'Exception caught during generation: {err}') from err

  async def generate(
    self,
    prompt: str | None,
    prompt_token_ids: list[int] | None = None,
    stop: str | t.Iterable[str] | None = None,
    stop_token_ids: list[int] | None = None,
    request_id: str | None = None,
    adapter_name: str | None = None,
    **attrs: t.Any,
  ) -> GenerationOutput:
    if stop is not None:
      attrs.update({'stop': stop})
    if stop_token_ids is not None:
      attrs.update({'stop_token_ids': stop_token_ids})
    config = self.config.model_construct_env(**attrs)
    texts, token_ids = [[]] * config['n'], [[]] * config['n']
    async for result in self.generate_iterator(
      prompt,
      prompt_token_ids=prompt_token_ids,
      request_id=request_id,
      adapter_name=adapter_name,
      **config.model_dump(),
    ):
      for output in result.outputs:
        texts[output.index].append(output.text)
        token_ids[output.index].extend(output.token_ids)
    if (final_result := result) is None:
      raise RuntimeError('No result is returned.')
    return final_result.model_copy(
      update=dict(
        prompt=prompt,
        outputs=[
          output.model_copy(update=dict(text=''.join(texts[output.index]), token_ids=token_ids[output.index]))
          for output in final_result.outputs
        ],
      )
    )
