from __future__ import annotations

import inspect, orjson, logging, dataclasses, bentoml, functools, attr, os, openllm_core, traceback, openllm, typing as t

from openllm_core.utils import (
  get_debug_mode,
  is_vllm_available,
  normalise_model_name,
  gen_random_uuid,
  dict_filter_none,
  Counter,
)
from openllm_core._typing_compat import LiteralQuantise, LiteralSerialisation, LiteralDtype, get_literal_args
from openllm_core._schemas import GenerationOutput

Dtype = t.Union[LiteralDtype, t.Literal['auto', 'half', 'float']]

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
  from vllm import AsyncEngineArgs, EngineArgs, RequestOutput


def check_engine_args(_, attr: attr.Attribute[dict[str, t.Any]], v: dict[str, t.Any]) -> dict[str, t.Any]:
  from vllm import AsyncEngineArgs

  fields = dataclasses.fields(AsyncEngineArgs)
  invalid_args = {k: v for k, v in v.items() if k not in {f.name for f in fields}}
  if len(invalid_args) > 0:
    raise ValueError(f'Invalid engine args: {list(invalid_args)}')
  return v


def check_quantization(_, attr: attr.Attribute[LiteralQuantise], v: str | None) -> LiteralQuantise | None:
  if v is not None and v not in get_literal_args(LiteralQuantise):
    raise ValueError(f'Invalid quantization method: {v}')
  return v


def update_engine_args(v: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
  env_json_string = os.environ.get('ENGINE_CONFIG', None)

  config_from_env = {}
  if env_json_string is not None:
    try:
      config_from_env = orjson.loads(env_json_string)
    except orjson.JSONDecodeError as e:
      raise RuntimeError("Failed to parse 'ENGINE_CONFIG' as valid JSON string.") from e
  config_from_env.update(v)
  return config_from_env


@attr.define(init=False)
class LLM:
  model_id: str
  bentomodel: bentoml.Model
  serialisation: LiteralSerialisation
  config: openllm_core.LLMConfig
  dtype: Dtype
  quantise: t.Optional[LiteralQuantise] = attr.field(default=None, validator=check_quantization)
  trust_remote_code: bool = attr.field(default=False)
  engine_args: t.Dict[str, t.Any] = attr.field(factory=dict, validator=check_engine_args, converter=update_engine_args)

  _mode: t.Literal['batch', 'async'] = attr.field(default='async', repr=False)
  _path: str = attr.field(
    init=False,
    default=attr.Factory(
      lambda self: self.bentomodel.path if self.bentomodel is not None else self.model_id, takes_self=True
    ),
  )
  _counter: Counter = attr.field(init=False, factory=lambda: Counter(), repr=False)

  def __init__(
    self, _: str = '', /, _internal: bool = False, _mode: t.Literal['batch', 'async'] = 'async', **kwargs: t.Any
  ) -> None:
    if not _internal:
      raise RuntimeError(
        'Cannot instantiate LLM directly in the new API. Make sure to use `openllm.LLM.from_model` instead.'
      )
    kwargs['mode'] = _mode
    self.__attrs_init__(**kwargs)

  def __attrs_post_init__(self):
    assert self._model  # Ensure the model is initialised.
    if self.config is None:
      self.config = openllm_core.AutoConfig.from_id(self.model_id, trust_remote_code=self.trust_remote_code)

  @classmethod
  def from_model(
    cls,
    model_id: str,
    dtype: str = 'auto',
    bentomodel: bentoml.Model | None = None,
    serialisation: LiteralSerialisation = 'safetensors',
    quantise: LiteralQuantise | None = None,
    trust_remote_code: bool = False,
    llm_config: openllm_core.LLMConfig | None = None,
    mode: t.Literal['batch', 'async'] = 'async',
    **engine_args: t.Any,
  ) -> LLM:
    return cls(
      _internal=True,
      _mode=mode,
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

  def _get_engine_args(self, mode: t.Literal['batch', 'async'] = 'async') -> AsyncEngineArgs | EngineArgs:
    if not is_vllm_available():
      raise RuntimeError("'vllm' is not available. Make sure to install with 'pip install vllm>0.4'")

    from vllm import EngineArgs, AsyncEngineArgs

    num_gpus, dev = 1, openllm.utils.device_count()
    if dev >= 2:
      num_gpus = min(dev // 2 * 2, dev)

    overriden_dict = {
      'tensor_parallel_size': num_gpus,
      'model': self._path,
      'tokenizer': self._path,
      'trust_remote_code': self.trust_remote_code,
      'dtype': self.dtype,
      'quantization': self.quantise,
    }
    if any(k in self.engine_args for k in overriden_dict.keys()):
      logger.warning(
        'The following key will be overriden by openllm: %s (got %s set)',
        list(overriden_dict),
        [k for k in overriden_dict if k in self.engine_args],
      )

    self.engine_args.update(overriden_dict)
    if 'worker_use_ray' not in self.engine_args:
      self.engine_args['worker_use_ray'] = False
    if 'tokenizer_mode' not in self.engine_args:
      self.engine_args['tokenizer_mode'] = 'auto'
    if 'disable_log_stats' not in self.engine_args:
      self.engine_args['disable_log_stats'] = not get_debug_mode()
    if 'gpu_memory_utilization' not in self.engine_args:
      self.engine_args['gpu_memory_utilization'] = 0.9

    if mode == 'async':
      if 'disable_log_requests' not in self.engine_args:
        self.engine_args['disable_log_requests'] = not get_debug_mode()
      if 'engine_use_ray' not in self.engine_args:
        self.engine_args['engine_use_ray'] = False

    return AsyncEngineArgs(**self.engine_args) if mode == 'async' else EngineArgs(**self.engine_args)

  @functools.cached_property
  def _model(self):
    if is_vllm_available():
      try:
        from vllm import AsyncLLMEngine, LLMEngine

        engine_cls = AsyncLLMEngine if self._mode == 'async' else LLMEngine
        return engine_cls.from_engine_args(self._get_engine_args(self._mode))
      except Exception as err:
        traceback.print_exc()
        raise openllm.exceptions.OpenLLMException(
          f'Failed to initialise vLLMEngine due to the following error:\n{err}'
        ) from err
    else:
      raise RuntimeError('Currently, OpenLLM is only supported running with a GPU and vLLM backend.')

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
    **attrs: t.Any,
  ) -> t.AsyncGenerator[RequestOutput, None]:
    if self._mode != 'async':
      raise RuntimeError(
        "'generate_iterator' is reserved only for online serving. For batch inference use 'LLM.batch' instead."
      )

    from vllm import SamplingParams, TextPrompt, TokensPrompt

    if prompt_token_ids is not None:
      inputs = TokensPrompt(prompt_token_ids=prompt_token_ids)
    else:
      if prompt is None:
        raise ValueError('Either "prompt" or "prompt_token_ids" must be passed.')
      inputs = TextPrompt(prompt=prompt)

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
        inputs=inputs,
        sampling_params=SamplingParams(**{
          k: config.__getitem__(k) for k in set(inspect.signature(SamplingParams).parameters.keys())
        }),
        request_id=request_id,
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
    **attrs: t.Any,
  ) -> GenerationOutput:
    if self._mode != 'async':
      raise RuntimeError(
        "'generate' is reserved only for online serving. For batch inference use 'LLM.batch' instead."
      )

    if stop is not None:
      attrs.update({'stop': stop})
    if stop_token_ids is not None:
      attrs.update({'stop_token_ids': stop_token_ids})
    config = self.config.model_construct_env(**attrs)
    async for result in self.generate_iterator(
      prompt, prompt_token_ids=prompt_token_ids, request_id=request_id, **config.model_dump()
    ):
      pass
    if (final_result := result) is None:
      raise RuntimeError('No result is returned.')
    return GenerationOutput.from_vllm(final_result).model_copy(update=dict(prompt=prompt))

  def batch(
    self,
    prompts: t.Union[str, t.List[str]],
    sampling_params: t.Optional[t.Union[t.Dict[str, t.Any], t.List[t.Dict[str, t.Any]]]] = None,
  ) -> t.List[GenerationOutput]:
    if self._mode != 'batch':
      raise RuntimeError(
        "'batch' is reserved for offline batch inference. For online serving use 'LLM.generate' or 'LLM.generate_iterator' instead."
      )

    from vllm import SamplingParams, TextPrompt

    if isinstance(prompts, str):
      prompts = [prompts]
    num_requests = len(prompts)

    if isinstance(sampling_params, list) and len(sampling_params) != num_requests:
      raise ValueError('Mismatch number of prompts and sampling params. (They must be the same length)')

    if sampling_params is None:
      configs = [self.config] * num_requests
    elif isinstance(sampling_params, dict):
      configs = [self.config.model_construct_env(**sampling_params)] * num_requests
    else:
      configs = [self.config.model_construct_env(**it) for it in sampling_params]

    for i in range(num_requests):
      request_id = str(next(self._counter))
      config = configs[i]
      self._model.add_request(
        request_id,
        TextPrompt(prompt=prompts[i]),
        SamplingParams(**{k: config.__getitem__(k) for k in set(inspect.signature(SamplingParams).parameters.keys())}),
      )

    # now run the engine
    outputs = []
    while self._model.has_unfinished_requests():
      step_outputs = self._model.step()
      for output in step_outputs:
        if output.finished:
          outputs.append(GenerationOutput.from_vllm(output))  # noqa
    outputs = sorted(outputs, key=lambda x: int(x.request_id))
    return outputs
