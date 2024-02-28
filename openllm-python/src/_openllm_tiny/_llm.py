from __future__ import annotations

import inspect, orjson, dataclasses, functools, bentoml, attr, openllm_core, traceback, openllm, typing as t
from openllm_core.utils import VersionInfo, check_bool_env, is_vllm_available, normalise_model_name, gen_random_uuid
from openllm_core._typing_compat import LiteralQuantise, LiteralSerialisation, LiteralDtype
from openllm_core._schemas import GenerationOutput, GenerationInput
from _bentoml_sdk.service import ServiceConfig

Dtype = t.Union[LiteralDtype, t.Literal['auto', 'half', 'float']]


def check_engine_args(_, attr: attr.Attribute[dict[str, t.Any]], v: dict[str, t.Any]) -> dict[str, t.Any]:
  from vllm import AsyncEngineArgs

  fields = dataclasses.fields(AsyncEngineArgs)
  invalid_args = {k: v for k, v in v.items() if k not in {f.name for f in fields}}
  if len(invalid_args) > 0:
    raise ValueError(f'Invalid engine args: {list(invalid_args)}')
  return v


@attr.define(init=False)
class LLM:
  model_id: str
  tag: str
  revision: str
  local: bool
  serialisation: LiteralSerialisation
  config: openllm_core.LLMConfig
  dtype: Dtype
  quantise: t.Optional[LiteralQuantise] = attr.field(default=None)
  trust_remote_code: bool = attr.field(default=False)
  engine_args: t.Dict[str, t.Any] = attr.field(factory=dict, validator=check_engine_args)
  service_config: t.Optional[ServiceConfig] = attr.field(factory=dict)

  _model: t.Any = attr.field(init=False)
  _path: str = attr.field(init=False)

  def __init__(self, _: str = '', /, _internal: bool = False, **kwargs: t.Any) -> None:
    if not _internal:
      raise RuntimeError(
        'Cannot instantiate LLM directly in the new API. Make sure to use `openllm.LLM.from_model` instead.'
      )
    self.__attrs_init__(**kwargs)

    self._path = self.bentomodel.path if self.local else self.model_id

    if is_vllm_available():
      num_gpus, dev = 1, openllm.utils.device_count()
      if dev >= 2:
        num_gpus = min(dev // 2 * 2, dev)
      quantise = self.quantise if self.quantise and self.quantise in {'gptq', 'awq', 'squeezellm'} else None
      dtype = 'float16' if quantise == 'gptq' else self.dtype  # NOTE: quantise GPTQ doesn't support bfloat16 yet.

      self.engine_args.setdefault('gpu_memory_utilization', 0.9)
      self.engine_args.update({
        'worker_use_ray': False,
        'engine_use_ray': False,
        'tokenizer_mode': 'auto',
        'tensor_parallel_size': num_gpus,
        'model': self._path,
        'tokenizer': self._path,
        'trust_remote_code': self.trust_remote_code,
        'dtype': dtype,
        'quantization': quantise,
      })
      try:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self._model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.engine_args))
      except Exception as err:
        traceback.print_exc()
        raise openllm.exceptions.OpenLLMException(
          f'Failed to initialise vLLMEngine due to the following error:\n{err}'
        ) from err
    else:
      raise RuntimeError('Currently, OpenLLM is only supported running with a GPU and vLLM backend.')

  @property
  def bentomodel(self) -> bentoml.Model:
    return bentoml.models.get(self.tag)

  @classmethod
  def from_model(
    cls,
    model: bentoml.Model,
    llm_config: openllm_core.LLMConfig | None = None,
    service_config: ServiceConfig | None = None,
    **engine_args: t.Any,
  ) -> LLM:
    metadata = model.info.metadata

    api_version: str | None = metadata.get('api_version')
    if api_version is None or VersionInfo.from_version_string(api_version) < VersionInfo.from_version_string('0.5.0'):
      raise RuntimeError(
        'Model API version is too old. Please update the model. Make sure to run openllm prune -y --include-bentos'
      )

    dtype = metadata['dtype']
    local = metadata['_local']
    model_id = metadata['model_id']
    trust_remote_code = check_bool_env('TRUST_REMOTE_CODE', default=metadata['trust_remote_code'])

    if llm_config is None:
      llm_config = openllm_core.AutoConfig.from_bentomodel(model)

    return cls(
      _internal=True,
      model_id=model_id,
      tag=str(model.tag),
      local=local,
      quantise=metadata.get('quantize'),
      serialisation=metadata['serialisation'],
      config=llm_config,
      dtype=dtype,
      revision=metadata['_revision'],
      engine_args=engine_args,
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
  def _tokenizerr(self):
    import transformers

    return transformers.AutoTokenizer.from_pretrained(self._path, trust_remote_code=self.trust_remote_code)

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
    from vllm import SamplingParams

    config = self.config.model_copy(update=attrs)

    stop_token_ids = stop_token_ids or []
    eos_token_id = attrs.get('eos_token_id', config['eos_token_id'])
    if eos_token_id and not isinstance(eos_token_id, list):
      eos_token_id = [eos_token_id]
    stop_token_ids.extend(eos_token_id or [])
    if (config_eos := config['eos_token_id']) and config_eos not in stop_token_ids:
      stop_token_ids.append(config_eos)
    if self._tokenizer.eos_token_id not in stop_token_ids:
      stop_token_ids.append(self._tokenizer.eos_token_id)
    if stop is None:
      stop = set()
    elif isinstance(stop, str):
      stop = {stop}
    else:
      stop = set(stop)

    request_id = gen_random_uuid() if request_id is None else request_id
    previous_texts, previous_num_tokens = [''] * config['n'], [0] * config['n']

    config = config.model_copy(update=dict(stop=list(stop), stop_token_ids=stop_token_ids))
    top_p = 1.0 if config['temperature'] <= 1e-5 else config['top_p']
    generation_config = config.generation_config.model_copy(update={'top_p': top_p})
    sampling_params = SamplingParams(**{
      k: getattr(generation_config, k, None) for k in set(inspect.signature(SamplingParams).parameters.keys())
    })

    try:
      generator = self._model.generate(
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
    *,
    _generated: GenerationInput | None = None,
    **attrs: t.Any,
  ) -> GenerationOutput:
    if stop is not None:
      attrs.update({'stop': stop})
    if stop_token_ids is not None:
      attrs.update({'stop_token_ids': stop_token_ids})
    config = self.config.model_copy(update=attrs)
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
