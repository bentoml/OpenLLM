from __future__ import annotations
import functools, logging, os, warnings, types, typing as t
from copy import deepcopy
import attr, orjson, bentoml, openllm_core, openllm
from openllm_core._schemas import (
  MessageParam,
  GenerationInput,
  GenerationInputDict,
  GenerationOutput,
  MessagesConverterInput,
)
from openllm_core._typing_compat import (
  AdapterTuple,
  AdapterType,
  LiteralBackend,
  LiteralDtype,
  LiteralQuantise,
  LiteralSerialisation,
  ParamSpec,
  Unpack,
  Concatenate,
  M,
  T,
)
from openllm_core.utils import (
  check_bool_env,
  codegen,
  first_not_none,
  flatten_attrs,
  gen_random_uuid,
  is_vllm_available,
  normalise_model_name,
)

from .exceptions import ForbiddenAttributeError

if t.TYPE_CHECKING:
  import torch, transformers
  from _bentoml_sdk.service import Service
  from peft.config import PeftConfig
  from openllm_core._configuration import LLMConfig
  from ._runners import Runner, DeprecatedRunner
  from ._llm import LLMService

logger = logging.getLogger(__name__)
_AdapterTuple: type[AdapterTuple] = codegen.make_attr_tuple_class('AdapterTuple', ['adapter_id', 'name', 'config'])
ResolvedAdapterMap = t.Dict[AdapterType, t.Dict[str, t.Tuple['PeftConfig', str]]]
P = ParamSpec('P')


@attr.define(slots=False, repr=False, init=False)
class LLM(t.Generic[M, T]):
  async def generate(
    self, prompt, prompt_token_ids=None, stop=None, stop_token_ids=None, request_id=None, adapter_name=None, **attrs
  ):
    if adapter_name is not None and self.__llm_backend__ != 'pt':
      raise NotImplementedError(f'Adapter is not supported with {self.__llm_backend__}.')
    config = self.config.model_construct_env(**attrs)
    texts, token_ids = [[]] * config['n'], [[]] * config['n']
    async for result in self.generate_iterator(
      prompt, prompt_token_ids, stop, stop_token_ids, request_id, adapter_name, **config.model_dump(flatten=True)
    ):
      for output in result.outputs:
        texts[output.index].append(output.text)
        token_ids[output.index].extend(output.token_ids)
    if (final_result := result) is None:
      raise RuntimeError('No result is returned.')
    return final_result.with_options(
      prompt=prompt,
      outputs=[
        output.with_options(text=''.join(texts[output.index]), token_ids=token_ids[output.index])
        for output in final_result.outputs
      ],
    )

  async def generate_iterator(
    self, prompt, prompt_token_ids=None, stop=None, stop_token_ids=None, request_id=None, adapter_name=None, **attrs
  ):
    if adapter_name is not None and self.__llm_backend__ != 'pt':
      raise NotImplementedError(f'Adapter is not supported with {self.__llm_backend__}.')
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

    for tid in stop_token_ids:
      if tid:
        stop.add(self.tokenizer.decode(tid))

    if prompt_token_ids is None:
      if prompt is None:
        raise ValueError('Either prompt or prompt_token_ids must be specified.')
      prompt_token_ids = self.tokenizer.encode(prompt)

    request_id = gen_random_uuid() if request_id is None else request_id
    previous_texts, previous_num_tokens = [''] * config['n'], [0] * config['n']
    try:
      generator = self.proxy(
        prompt_token_ids, request_id, stop=list(stop), adapter_name=adapter_name, **config.model_dump(flatten=True)
      )
    except Exception as err:
      raise RuntimeError(f'Failed to start generation task: {err}') from err

    try:
      async for out in generator:
        if not self._eager:
          generated = GenerationOutput.from_runner(out).with_options(prompt=prompt)
        else:
          generated = out.with_options(prompt=prompt)
        delta_outputs = [None] * len(generated.outputs)
        for output in generated.outputs:
          i = output.index
          delta_tokens, delta_text = output.token_ids[previous_num_tokens[i] :], output.text[len(previous_texts[i]) :]
          previous_texts[i], previous_num_tokens[i] = output.text, len(output.token_ids)
          delta_outputs[i] = output.with_options(text=delta_text, token_ids=delta_tokens)
        yield generated.with_options(outputs=delta_outputs)
    except Exception as err:
      raise RuntimeError(f'Exception caught during generation: {err}') from err

  @classmethod
  def from_pretrained(
    cls,
    model_id,
    /,
    *decls,
    bentomodel_tag=None,
    bentomodel_version=None,
    quantize=None,
    quantization_config=None,
    backend=None,
    dtype='auto',
    serialisation='safetensors',
    trust_remote_code=False,
    low_cpu_mem_usage=True,
    llm_config=None,
    max_model_len=None,
    gpu_memory_utilization=0.9,
    _internal=False,
    **attrs,
  ):
    if not _internal:
      raise RuntimeError(
        "'from_pretrained' is an internal API and not to be use. Make sure to use 'openllm.prepare_model' and 'LLM.from_model' instead."
      )
    att = deepcopy(attrs)
    return cls.from_model(
      openllm.prepare_model(
        model_id,
        *decls,
        bentomodel_tag=bentomodel_tag,
        bentomodel_version=bentomodel_version,
        quantize=quantize,
        quantization_config=quantization_config,
        backend=backend,
        dtype=dtype,
        serialisation=serialisation,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **attrs,
      ),
      llm_config=llm_config,
      max_model_len=max_model_len,
      gpu_memory_utilization=gpu_memory_utilization,
      **att,
    )

  @classmethod
  def from_model(
    cls,
    model,
    backend=None,
    llm_config=None,
    max_model_len=None,
    gpu_memory_utilization=0.9,
    trust_remote_code=False,
    service_config=None,
    **attrs,
  ):
    if not isinstance(model, bentoml.Model):
      raise TypeError(f'Expected `bentoml.Model`, got {type(model)}')
    return cls(
      model.info.metadata['model_id'],
      model_version=model.tag.version,
      model_tag=model.tag,
      llm_config=llm_config,
      backend=first_not_none(backend, default=model.info.metadata['backend']),
      quantize=model.info.metadata.get('_quantize'),
      serialisation=model.info.metadata['serialisation'],
      trust_remote_code=trust_remote_code,
      max_model_len=max_model_len,
      gpu_memory_utilization=gpu_memory_utilization,
      service_config=service_config,
      _local=model.info.metadata['_local'],
      _eager=True,
      _from_model=True,
      _bentomodel=model,
      **attrs,
    )

  # NOTE: If you are here to see how generate_iterator and generate works, see above.
  # The below are mainly for internal implementation that you don't have to worry about.
  _model_id: str
  _revision: t.Optional[str]  #
  _quantization_config: t.Optional[
    t.Union[transformers.BitsAndBytesConfig, transformers.GPTQConfig, transformers.AwqConfig]
  ]
  _quantise: t.Optional[LiteralQuantise]
  _model_decls: t.Tuple[t.Any, ...]
  __model_attrs: t.Dict[str, t.Any]  #
  __tokenizer_attrs: t.Dict[str, t.Any]
  _tag: bentoml.Tag
  _serialisation: LiteralSerialisation
  _local: bool
  _max_model_len: t.Optional[int]  #
  _gpu_memory_utilization: float
  service_config: t.Dict[str, t.Any] = attr.field(factory=dict)
  _eager: bool = False  # whether to load the model eagerly

  __llm_dtype__: t.Union[LiteralDtype, t.Literal['auto', 'half', 'float']] = 'auto'
  __llm_torch_dtype__: 'torch.dtype' = None
  __llm_config__: t.Optional[LLMConfig] = None
  __llm_backend__: LiteralBackend = None
  __llm_quantization_config__: t.Optional[
    t.Union[transformers.BitsAndBytesConfig, transformers.GPTQConfig, transformers.AwqConfig]
  ] = None
  __llm_inference_bases__: t.Optional[Runner[M, T]] = None
  __llm_runner__: t.Optional[DeprecatedRunner[M, T]] = None
  __llm_model__: t.Optional[M] = None
  __llm_service__: t.Optional[Service[LLMService]] = None
  __llm_tokenizer__: t.Optional[T] = None
  __llm_trust_remote_code__: bool = False
  __llm_proxy_caller__: t.Callable[
    Concatenate[t.List[int], str, t.Optional[t.Union[t.Iterable[str], str]], t.Optional[str], t.Optional[str], P],
    t.AsyncGenerator[t.Union[str, GenerationOutput], None],
  ] = None

  def __init__(
    self,
    model_id,
    model_version=None,
    model_tag=None,
    llm_config=None,
    backend=None,
    *args,
    quantize=None,
    quantization_config=None,
    serialisation='safetensors',
    trust_remote_code=False,
    dtype='auto',
    low_cpu_mem_usage=True,
    max_model_len=None,
    gpu_memory_utilization=0.9,
    service_config=None,
    embedded=False,
    _local=False,
    _eager=False,
    _from_model=False,
    _bentomodel=None,
    **attrs,
  ):
    if service_config is None:
      service_config = {}
    if not _from_model:
      logger.warning('This is an internal API. Please use `openllm.LLM.from_model` instead.')
      att = deepcopy(attrs)
      _bentomodel = openllm.prepare_model(
        model_id,
        *args,
        bentomodel_tag=model_tag,
        bentomodel_version=model_version,
        quantize=quantize,
        quantization_config=quantization_config,
        backend=backend,
        dtype=dtype,
        serialisation=serialisation,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=low_cpu_mem_usage,
        **att,
      )
      if model_tag is None:
        model_tag = _bentomodel.tag
      if model_version is None:
        model_version = _bentomodel.tag.version

    attrs.update({'low_cpu_mem_usage': low_cpu_mem_usage})
    # parsing tokenizer and model kwargs, as the hierarchy is param pass > default
    model_attrs, tokenizer_attrs = flatten_attrs(**attrs)

    self.__attrs_init__(
      model_id=model_id,
      revision=model_version,
      tag=bentoml.Tag.from_taglike(model_tag),
      quantization_config=quantization_config,
      quantise=getattr(self._Quantise, backend)(self, quantize),
      model_decls=args,
      serialisation=serialisation,
      local=_local,
      max_model_len=max_model_len,
      gpu_memory_utilization=gpu_memory_utilization,
      service_config=service_config,
      eager=_eager,
      LLM__model_attrs=model_attrs,
      LLM__tokenizer_attrs=tokenizer_attrs,
      llm_dtype__=dtype.lower(),
      llm_backend__=backend,
      llm_config__=llm_config,
      llm_trust_remote_code__=trust_remote_code,
    )
    self._tag = _bentomodel.tag  # resolve the tag
    self._revision = _bentomodel.tag.version
    if _eager and embedded:
      raise RuntimeError('Embedded mode is not supported when eager loading.')
    if _eager:
      self.bases()
    elif embedded:
      logger.warning('NOT RECOMMENDED in production and SHOULD ONLY used for development.')
      self.runner.init_local(quiet=True)

  class _Quantise:
    @staticmethod
    def pt(llm: LLM, quantise=None):
      return quantise

    @staticmethod
    def vllm(llm: LLM, quantise=None):
      return quantise

    @staticmethod
    def ctranslate(llm: LLM, quantise=None):
      if quantise in {'int4', 'awq', 'gptq', 'squeezellm'}:
        raise ValueError(f"Quantisation '{quantise}' is not supported for backend 'ctranslate'")
      if quantise == 'int8':
        quantise = 'int8_float16' if llm._has_gpus else 'int8_float32'
      return quantise

  @functools.cached_property
  def _has_gpus(self):
    try:
      from cuda import cuda

      err, *_ = cuda.cuInit(0)
      if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError('Failed to initialise CUDA runtime binding.')
      err, _ = cuda.cuDeviceGetCount()
      if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError('Failed to get CUDA device count.')
      return True
    except (ImportError, RuntimeError):
      return False

  @property
  def _torch_dtype(self):
    import torch
    from .serialisation.transformers import _torch_dtype

    if not isinstance(self.__llm_torch_dtype__, torch.dtype):
      self.__llm_torch_dtype__ = _torch_dtype(self.__llm_dtype__, self.model_id, self.trust_remote_code)
    return self.__llm_torch_dtype__

  @property
  def proxy(self):
    if self.__llm_proxy_caller__ is None:
      if self._eager:
        self.__llm_proxy_caller__ = self.bases().generate_iterator
      else:
        from bentoml._internal.runner.runner_handle import DummyRunnerHandle

        if isinstance(self.runner._runner_handle, DummyRunnerHandle):
          if os.getenv('BENTO_PATH') is not None:
            raise RuntimeError('Runner client failed to set up correctly.')
          else:
            self.runner.init_local(quiet=True)
        self.__llm_proxy_caller__ = self.runner.generate_iterator.async_stream
    return self.__llm_proxy_caller__

  def _cascade_backend(self) -> LiteralBackend:
    logger.warning(
      'It is recommended to specify the backend explicitly. Cascading backend might lead to unexpected behaviour.'
    )
    if self._has_gpus:
      return 'vllm' if is_vllm_available() else 'pt'
    else:
      return 'pt'

  def __setattr__(self, attr, value):
    if attr in {'model', 'tokenizer', 'runner', 'import_kwargs'}:
      raise ForbiddenAttributeError(f'{attr} should not be set during runtime.')
    super().__setattr__(attr, value)

  def __del__(self):
    try:
      del (
        self.__llm_model__,
        self.__llm_tokenizer__,
        self.__llm_runner__,
        self.__llm_inference_bases__,
        self.__llm_proxy_caller__,
      )
    except AttributeError:
      pass

  def __repr__(self) -> str:
    return f'{self.__class__.__name__} {orjson.dumps({k: v for k, v in (("model_id", self._model_id if not self._local else self.tag.name), ("revision", self._revision if self._revision else self.tag.version), ("backend", self.__llm_backend__), ("type", self.llm_type))}, option=orjson.OPT_INDENT_2).decode()}'

  @property
  def model_id(self):
    return self._model_id

  @property
  def revision(self):
    return self._revision

  @property
  def tag(self):
    return self._tag

  @property
  def bentomodel(self):
    return bentoml.models.get(self.tag)

  @property
  def local(self):
    return self._local

  @property
  def quantise(self):
    return self._quantise

  @property
  def llm_type(self):
    return normalise_model_name(self._model_id)

  @property
  def llm_parameters(self):
    return (self._model_decls, self._model_attrs), self._tokenizer_attrs

  @property
  def _model_attrs(self):
    return {**self.import_kwargs[0], **self.__model_attrs}

  @_model_attrs.setter
  def _model_attrs(self, model_attrs):
    self.__model_attrs = model_attrs

  @property
  def _tokenizer_attrs(self):
    return {**self.import_kwargs[1], **self.__tokenizer_attrs}

  @property
  def import_kwargs(self):
    return {'device_map': 'auto' if self._has_gpus else None, 'torch_dtype': self._torch_dtype}, {
      'padding_side': 'left',
      'truncation_side': 'left',
    }

  @property
  def trust_remote_code(self):
    env = os.getenv('TRUST_REMOTE_CODE')
    if env is not None:
      return check_bool_env('TRUST_REMOTE_CODE', env)
    return self.__llm_trust_remote_code__

  @property
  def quantization_config(self):
    if self.__llm_quantization_config__ is None:
      from ._quantisation import infer_quantisation_config

      if self._quantization_config is not None:
        self.__llm_quantization_config__ = self._quantization_config
      elif self._quantise is not None:
        self.__llm_quantization_config__, self._model_attrs = infer_quantisation_config(
          self, self._quantise, **self._model_attrs
        )
      else:
        raise ValueError("Either 'quantization_config' or 'quantise' must be specified.")
    return self.__llm_quantization_config__

  @property
  def identifying_params(self):
    return {
      'configuration': self.config.model_dump_json().decode(),
      'model_ids': orjson.dumps(self.config['model_ids']).decode(),
      'model_id': self.model_id,
    }

  @property
  def tokenizer(self):
    if self.__llm_tokenizer__ is None:
      self.__llm_tokenizer__ = openllm.serialisation.load_tokenizer(self, **self.llm_parameters[-1])
    return self.__llm_tokenizer__

  @property
  def runner(self):
    warnings.warn(
      'Using "llm.runner" only works with older bentoml.Service. Once upgrading to BentoML 1.2 and onwards make sure to use the new syntax.',
      DeprecationWarning,
      stacklevel=3,
    )
    from ._runners import runner

    if self.__llm_runner__ is None:
      self.__llm_runner__ = runner(self)
    return self.__llm_runner__

  @property
  def service(self) -> Service[LLMService]:
    if self.__llm_service__ is None:
      from ._runners import apis

      self.__llm_service__ = openllm.mount_entrypoints(
        bentoml.service(
          apis(self, GenerationInput.from_llm_config(self.config)),
          name=f'llm-{self.config["start-name"]}-service',
          **self.service_config,
        ),
        self,
      )

    return self.__llm_service__

  def bases(self) -> Runner[M, T]:
    from ._runners import bases

    if self.__llm_inference_bases__ is None:
      self.__llm_inference_bases__ = bases(self)()
    return self.__llm_inference_bases__

  @property
  def model(self):
    if self.__llm_model__ is None:
      self.__llm_model__ = openllm.serialisation.load_model(self, *self._model_decls, **self._model_attrs)
    return self.__llm_model__

  @property
  def config(self):
    if self.__llm_config__ is None:
      self.__llm_config__ = openllm_core.AutoConfig.from_llm(self).model_construct_env(**self._model_attrs)
    return self.__llm_config__


def apis(llm, generations):
  @openllm.utils.api(route='/v1/generate', output=GenerationOutput, input=generations)
  async def generate_v1(self, **parameters: Unpack[GenerationInputDict]) -> GenerationOutput:
    structured = generations.from_dict(**parameters)
    return await self.llm.generate(
      structured.prompt,
      structured.prompt_token_ids,  #
      structured.stop,
      structured.stop_token_ids,  #
      structured.request_id,
      structured.adapter_name,  #
      **structured.llm_config.model_dump(flatten=True),
    )

  @openllm.utils.api(route='/v1/generate_stream', input=generations)
  async def generate_stream_v1(self, **parameters: Unpack[GenerationInputDict]) -> t.AsyncGenerator[str, None]:
    structured = GenerationInput(**parameters)
    async for generated in self.llm.generate_iterator(
      structured.prompt,
      structured.prompt_token_ids,  #
      structured.stop,
      structured.stop_token_ids,  #
      structured.request_id,
      structured.adapter_name,  #
      **structured.llm_config.model_dump(flatten=True),
    ):
      yield f'data: {generated.model_dump_json()}\n\n'
    yield 'data: [DONE]\n\n'

  @openllm.utils.api(output=openllm.MetadataOutput, route='/v1/metadata')
  def metadata_v1(self) -> openllm.MetadataOutput:
    return openllm.MetadataOutput(
      timeout=self.llm.config['timeout'],
      model_name=self.llm.config['model_name'],
      backend=self.llm.__llm_backend__,
      model_id=self.llm.model_id,
      configuration=self.llm.config.model_dump_json().decode(),
    )

  @openllm.utils.api(route='/v1/helpers/messages')
  def helpers_messages_v1(
    self,
    message: MessagesConverterInput = MessagesConverterInput(
      add_generation_prompt=False,
      messages=[
        MessageParam(role='system', content='You are acting as Ernest Hemmingway.'),
        MessageParam(role='user', content='Hi there!'),
        MessageParam(role='assistant', content='Yes?'),
      ],
    ),
  ) -> str:
    add_generation_prompt, messages = message['add_generation_prompt'], message['messages']
    return self.llm.tokenizer.apply_chat_template(
      messages, add_generation_prompt=add_generation_prompt, tokenize=False
    )

  list_apis = {
    k: v for k, v in locals().items() if hasattr(v, 'func') and getattr(v.func, '__openllm_api_func__', False)
  }

  return types.new_class(
    llm.config.__class__.__name__[:-6] + 'Service',
    exec_body=lambda ns: ns.update({'llm': llm, '__module__': __name__, **list_apis}),
  )
