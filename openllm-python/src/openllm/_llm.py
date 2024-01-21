from __future__ import annotations
import functools, logging, os, warnings, typing as t
from copy import deepcopy
import attr, orjson, bentoml, openllm
from openllm_core._schemas import GenerationOutput
from openllm_core._typing_compat import (
  AdapterMap,
  AdapterTuple,
  AdapterType,
  LiteralBackend,
  LiteralDtype,
  LiteralQuantise,
  LiteralSerialisation,
  ParamSpec,
  Concatenate,
  M,
  T,
)
from openllm_core.exceptions import MissingDependencyError
from openllm_core.utils import (
  DEBUG,
  check_bool_env,
  codegen,
  first_not_none,
  flatten_attrs,
  gen_random_uuid,
  is_ctranslate_available,
  is_peft_available,
  is_vllm_available,
  normalise_model_name,
)

from .exceptions import ForbiddenAttributeError, OpenLLMException
from .serialisation.constants import PEFT_CONFIG_NAME

if t.TYPE_CHECKING:
  import torch, transformers
  from peft.config import PeftConfig
  from openllm_core._configuration import LLMConfig
  from ._runners import Runner, DeprecatedRunner

logger = logging.getLogger(__name__)
_AdapterTuple: type[AdapterTuple] = codegen.make_attr_tuple_class('AdapterTuple', ['adapter_id', 'name', 'config'])
ResolvedAdapterMap = t.Dict[AdapterType, t.Dict[str, t.Tuple['PeftConfig', str]]]
P = ParamSpec('P')


@attr.define(slots=False, repr=False, init=False)
class LLM(t.Generic[M, T]):
  async def generate(self, prompt, prompt_token_ids=None, stop=None, stop_token_ids=None, request_id=None, adapter_name=None, **attrs):
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
      outputs=[output.with_options(text=''.join(texts[output.index]), token_ids=token_ids[output.index]) for output in final_result.outputs],
    )

  async def generate_iterator(self, prompt, prompt_token_ids=None, stop=None, stop_token_ids=None, request_id=None, adapter_name=None, **attrs):
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
      generator = self.proxy(prompt_token_ids, request_id, stop=list(stop), adapter_name=adapter_name, **config.model_dump(flatten=True))
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
    adapter_map=None,
    _internal=False,
    **attrs,
  ):
    if not _internal: raise RuntimeError("'from_pretrained' is an internal API and not to be use. Make sure to use 'openllm.prepare_model' and 'LLM.from_model' instead.")
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
      adapter_map=adapter_map,
      **att,
    )

  @classmethod
  def from_model(
    cls, model, backend=None, llm_config=None, max_model_len=None, gpu_memory_utilization=0.9, trust_remote_code=False, adapter_map=None, **attrs
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
      adapter_map=adapter_map,
      serialisation=model.info.metadata['serialisation'],
      trust_remote_code=trust_remote_code,
      max_model_len=max_model_len,
      gpu_memory_utilization=gpu_memory_utilization,
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
  _quantization_config: t.Optional[t.Union[transformers.BitsAndBytesConfig, transformers.GPTQConfig, transformers.AwqConfig]]
  _quantise: t.Optional[LiteralQuantise]
  _model_decls: t.Tuple[t.Any, ...]
  __model_attrs: t.Dict[str, t.Any]  #
  __tokenizer_attrs: t.Dict[str, t.Any]
  _tag: bentoml.Tag
  _adapter_map: t.Optional[AdapterMap]  #
  _serialisation: LiteralSerialisation
  _local: bool
  _max_model_len: t.Optional[int]  #
  _gpu_memory_utilization: float
  _eager: bool = False  # whether to load the model eagerly

  __llm_dtype__: t.Union[LiteralDtype, t.Literal['auto', 'half', 'float']] = 'auto'
  __llm_torch_dtype__: 'torch.dtype' = None
  __llm_config__: t.Optional[LLMConfig] = None
  __llm_backend__: LiteralBackend = None
  __llm_quantization_config__: t.Optional[t.Union[transformers.BitsAndBytesConfig, transformers.GPTQConfig, transformers.AwqConfig]] = None
  __llm_inference_bases__: t.Optional[Runner[M, T]] = None
  __llm_runner__: t.Optional[DeprecatedRunner[M, T]] = None
  __llm_model__: t.Optional[M] = None
  __llm_tokenizer__: t.Optional[T] = None
  __llm_adapter_map__: t.Optional[ResolvedAdapterMap] = None
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
    adapter_map=None,
    serialisation='safetensors',
    trust_remote_code=False,
    dtype='auto',
    low_cpu_mem_usage=True,
    max_model_len=None,
    gpu_memory_utilization=0.9,
    embedded=False,
    _local=False,
    _eager=False,
    _from_model=False,
    _bentomodel=None,
    **attrs,
  ):
    if not _from_model:
      logger.warning('This is an internal API. Please use `openllm.LLM.from_model` instead.')

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
      adapter_map=convert_peft_config_type(adapter_map) if adapter_map is not None else None,
      serialisation=serialisation,
      local=_local,
      max_model_len=max_model_len,
      gpu_memory_utilization=gpu_memory_utilization,
      eager=_eager,
      LLM__model_attrs=model_attrs,
      LLM__tokenizer_attrs=tokenizer_attrs,
      llm_dtype__=dtype.lower(),
      llm_backend__=backend,
      llm_config__=llm_config,
      llm_trust_remote_code__=trust_remote_code,
    )
    if _bentomodel is None: _bentomodel = bentoml.models.get(model_tag)
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
      self.__llm_torch_dtype_ = _torch_dtype(self.__llm_dtype__, self.model_id, self.trust_remote_code)
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
    logger.warning('It is recommended to specify the backend explicitly. Cascading backend might lead to unexpected behaviour.')
    if self._has_gpus:
      if is_vllm_available():
        return 'vllm'
      elif is_ctranslate_available():
        return 'ctranslate'
    elif is_ctranslate_available():
      return 'ctranslate'
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
        self.__llm_adapter_map__,
        self.__llm_runner__,
        self.__llm_inference_bases__,
        self.__llm_proxy_caller__,
      )
    except AttributeError:
      pass

  def __repr__(self) -> str:
    return f'{self.__class__.__name__} {orjson.dumps({k: v for k, v in (("model_id", self._model_id if not self._local else self.tag.name), ("revision", self._revision if self._revision else self.tag.version), ("backend", self.__llm_backend__), ("type", self.llm_type))}, option=orjson.OPT_INDENT_2).decode()}'

  @property
  def model_id(self): return self._model_id
  @property
  def revision(self): return self._revision
  @property
  def tag(self): return self._tag
  @property
  def bentomodel(self): return bentoml.models.get(self.tag)
  @property
  def has_adapters(self): return self._adapter_map is not None
  @property
  def local(self): return self._local
  @property
  def quantise(self): return self._quantise
  @property
  def llm_type(self): return normalise_model_name(self._model_id)
  @property
  def llm_parameters(self): return (self._model_decls, self._model_attrs), self._tokenizer_attrs
  @property
  def _model_attrs(self): return {**self.import_kwargs[0], **self.__model_attrs}
  @_model_attrs.setter
  def _model_attrs(self, model_attrs): self.__model_attrs = model_attrs
  @property
  def _tokenizer_attrs(self): return {**self.import_kwargs[1], **self.__tokenizer_attrs}
  @property
  def import_kwargs(self): return {'device_map': 'auto' if self._has_gpus else None, 'torch_dtype': self._torch_dtype}, {'padding_side': 'left', 'truncation_side': 'left'}

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
        self.__llm_quantization_config__, self._model_attrs = infer_quantisation_config(self, self._quantise, **self._model_attrs)
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

  def bases(self) -> Runner[M, T]:
    from ._runners import bases

    if self.__llm_inference_bases__ is None:
      self.__llm_inference_bases__ = bases(self)()
    return self.__llm_inference_bases__

  def prepare(self, adapter_type='lora', use_gradient_checking=True, **attrs):
    if self.__llm_backend__ != 'pt':
      raise RuntimeError('Fine tuning is only supported for PyTorch backend.')
    from peft.mapping import get_peft_model
    from peft.utils.other import prepare_model_for_kbit_training

    model = get_peft_model(
      prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=use_gradient_checking),
      self.config['fine_tune_strategies'].get(adapter_type, self.config.make_fine_tune_config(adapter_type)).train().with_config(**attrs).build(),
    )
    if DEBUG:
      model.print_trainable_parameters()
    return model, self.tokenizer

  def prepare_for_training(self, *args, **attrs):
    warnings.warn('`prepare_for_training` is deprecated and will be removed in the future. Use `prepare` instead.', DeprecationWarning, stacklevel=3)
    return self.prepare(*args, **attrs)

  @property
  def adapter_map(self):
    if not is_peft_available():
      raise MissingDependencyError("Failed to import 'peft'. Make sure to do 'pip install \"openllm[fine-tune]\"'")
    if not self.has_adapters:
      raise AttributeError('Adapter map is not available.')
    assert self._adapter_map is not None
    if self.__llm_adapter_map__ is None:
      _map: ResolvedAdapterMap = {k: {} for k in self._adapter_map}
      for adapter_type, adapter_tuple in self._adapter_map.items():
        base = first_not_none(self.config['fine_tune_strategies'].get(adapter_type), default=self.config.make_fine_tune_config(adapter_type))
        for adapter in adapter_tuple:
          _map[adapter_type][adapter.name] = (base.with_config(**adapter.config).build(), adapter.adapter_id)
      self.__llm_adapter_map__ = _map
    return self.__llm_adapter_map__

  @property
  def model(self):
    if self.__llm_model__ is None:
      self.__llm_model__ = openllm.serialisation.load_model(self, *self._model_decls, **self._model_attrs)
    return self.__llm_model__

  @property
  def config(self):
    import transformers

    if self.__llm_config__ is None:
      if self.__llm_backend__ == 'ctranslate':
        try:
          hf_config = transformers.AutoConfig.from_pretrained(self.bentomodel.path_of('/hf'), trust_remote_code=self.trust_remote_code)
        except OpenLLMException:
          hf_config = transformers.AutoConfig.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        for architecture in hf_config.architectures:
          if architecture in openllm.AutoConfig._CONFIG_MAPPING_NAMES_TO_ARCHITECTURE():
            config = openllm.AutoConfig.infer_class_from_name(
              openllm.AutoConfig._CONFIG_MAPPING_NAMES_TO_ARCHITECTURE()[architecture]
            ).model_construct_env(**self._model_attrs)
            break
          else:
            raise OpenLLMException(
              f"Failed to infer the configuration class. Make sure the model is a supported model. Supported models are: {', '.join(openllm.AutoConfig._CONFIG_MAPPING_NAMES_TO_ARCHITECTURE.keys())}"
            )
      else:
        config = openllm.AutoConfig.infer_class_from_llm(self).model_construct_env(**self._model_attrs)
      self.__llm_config__ = config
    return self.__llm_config__


def convert_peft_config_type(adapter_map: dict[str, str]) -> AdapterMap:
  if not is_peft_available():
    raise RuntimeError("LoRA adapter requires 'peft' to be installed. Make sure to do 'pip install \"openllm[fine-tune]\"'")
  from huggingface_hub import hf_hub_download

  resolved: AdapterMap = {}
  for path_or_adapter_id, name in adapter_map.items():
    if name is None:
      raise ValueError('Adapter name must be specified.')
    if os.path.isfile(os.path.join(path_or_adapter_id, PEFT_CONFIG_NAME)):
      config_file = os.path.join(path_or_adapter_id, PEFT_CONFIG_NAME)
    else:
      try:
        config_file = hf_hub_download(path_or_adapter_id, PEFT_CONFIG_NAME)
      except Exception as err:
        raise ValueError(f"Can't find '{PEFT_CONFIG_NAME}' at '{path_or_adapter_id}'") from err
    with open(config_file, 'r') as file:
      resolved_config = orjson.loads(file.read())
    _peft_type = resolved_config['peft_type'].lower()
    if _peft_type not in resolved:
      resolved[_peft_type] = ()
    resolved[_peft_type] += (_AdapterTuple((path_or_adapter_id, name, resolved_config)),)
  return resolved
