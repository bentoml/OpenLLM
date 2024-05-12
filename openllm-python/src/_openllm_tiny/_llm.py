from __future__ import annotations

import inspect, orjson, dataclasses, bentoml, functools, attr, openllm_core, traceback, openllm, typing as t

from openllm_core.utils import (
  get_debug_mode,
  is_vllm_available,
  normalise_model_name,
  gen_random_uuid,
  dict_filter_none,
)
from openllm_core._typing_compat import LiteralQuantise, LiteralSerialisation, LiteralDtype
from openllm_core._schemas import GenerationOutput

Dtype = t.Union[LiteralDtype, t.Literal['auto', 'half', 'float']]

if t.TYPE_CHECKING:
  from vllm import RequestOutput


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
  bentomodel: bentoml.Model
  serialisation: LiteralSerialisation
  config: openllm_core.LLMConfig
  dtype: Dtype
  quantise: t.Optional[LiteralQuantise] = attr.field(default=None)
  trust_remote_code: bool = attr.field(default=False)
  engine_args: t.Dict[str, t.Any] = attr.field(factory=dict, validator=check_engine_args)

  _path: str = attr.field(
    init=False,
    default=attr.Factory(
      lambda self: self.bentomodel.path if self.bentomodel is not None else self.model_id, takes_self=True
    ),
  )

  def __init__(self, _: str = '', /, _internal: bool = False, **kwargs: t.Any) -> None:
    if not _internal:
      raise RuntimeError(
        'Cannot instantiate LLM directly in the new API. Make sure to use `openllm.LLM.from_model` instead.'
      )
    self.__attrs_init__(**kwargs)

  def __attrs_post_init__(self):
    assert self._model  # Ensure the model is initialised.

  @functools.cached_property
  def _model(self):
    if is_vllm_available():
      num_gpus, dev = 1, openllm.utils.device_count()
      if dev >= 2:
        num_gpus = min(dev // 2 * 2, dev)
      quantise = self.quantise if self.quantise and self.quantise in {'gptq', 'awq', 'squeezellm'} else None
      dtype = 'float16' if quantise == 'gptq' else self.dtype  # NOTE: quantise GPTQ doesn't support bfloat16 yet.

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
      if 'disable_log_stats' not in self.engine_args:
        self.engine_args['disable_log_stats'] = not get_debug_mode()
      if 'disable_log_requests' not in self.engine_args:
        self.engine_args['disable_log_requests'] = not get_debug_mode()
      if 'gpu_memory_utilization' not in self.engine_args:
        self.engine_args['gpu_memory_utilization'] = 0.9

      try:
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        return AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**self.engine_args))
      except Exception as err:
        traceback.print_exc()
        raise openllm.exceptions.OpenLLMException(
          f'Failed to initialise vLLMEngine due to the following error:\n{err}'
        ) from err
    else:
      raise RuntimeError('Currently, OpenLLM is only supported running with a GPU and vLLM backend.')

  @classmethod
  def from_model(
    cls,
    model_id: str,
    dtype: str,
    bentomodel: bentoml.Model | None = None,
    serialisation: LiteralSerialisation = 'safetensors',
    quantise: LiteralQuantise | None = None,
    trust_remote_code: bool = False,
    llm_config: openllm_core.LLMConfig | None = None,
    **engine_args: t.Any,
  ) -> LLM:
    return cls(
      _internal=True,
      model_id=model_id,
      bentomodel=bentomodel,
      quantise=quantise,
      serialisation=serialisation,
      config=llm_config,
      dtype=dtype,
      engine_args=engine_args,
      trust_remote_code=trust_remote_code,
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
  def _tokenizer(self):
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
  ) -> t.AsyncGenerator[RequestOutput, None]:
    from vllm import SamplingParams

    config = self.config.model_construct_env(**dict_filter_none(attrs))

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

    top_p = 1.0 if config['temperature'] <= 1e-5 else config['top_p']
    config = config.model_copy(update=dict(stop=list(stop), stop_token_ids=stop_token_ids, top_p=top_p))

    try:
      async for generations in self._model.generate(
        prompt,
        sampling_params=SamplingParams(**{
          k: config.__getitem__(k) for k in set(inspect.signature(SamplingParams).parameters.keys())
        }),
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
      ):
        yield generations
    except Exception as err:
      raise RuntimeError(f'Failed to start generation task: {err}') from err

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
    async for result in self.generate_iterator(
      prompt,
      prompt_token_ids=prompt_token_ids,
      request_id=request_id,
      adapter_name=adapter_name,
      **config.model_dump(),
    ):
      pass
    if (final_result := result) is None:
      raise RuntimeError('No result is returned.')
    return GenerationOutput.from_vllm(final_result).model_copy(update=dict(prompt=prompt))
