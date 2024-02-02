from __future__ import annotations
import functools, logging, os, warnings, typing as t
import attr, inflection, orjson, bentoml, openllm
from openllm_core._schemas import GenerationOutput
from openllm_core._typing_compat import (
  AdapterMap,
  AdapterTuple,
  AdapterType,
  LiteralBackend,
  LiteralDtype,
  LiteralQuantise,
  LiteralSerialisation,
  M,
  T,
)
from openllm_core.exceptions import MissingDependencyError
from openllm_core.utils import (
  DEBUG,
  apply,
  check_bool_env,
  codegen,
  first_not_none,
  flatten_attrs,
  gen_random_uuid,
  generate_hash_from_file,
  getenv,
  is_ctranslate_available,
  is_peft_available,
  is_vllm_available,
  resolve_filepath,
  validate_is_path,
)

from .exceptions import ForbiddenAttributeError, OpenLLMException
from .serialisation.constants import PEFT_CONFIG_NAME

if t.TYPE_CHECKING:
  import torch, transformers
  from peft.config import PeftConfig
  from openllm_core._configuration import LLMConfig
  from ._runners import Runner

logger = logging.getLogger(__name__)
_AdapterTuple: type[AdapterTuple] = codegen.make_attr_tuple_class('AdapterTuple', ['adapter_id', 'name', 'config'])
ResolvedAdapterMap = t.Dict[AdapterType, t.Dict[str, t.Tuple['PeftConfig', str]]]


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
    from bentoml._internal.runner.runner_handle import DummyRunnerHandle

    if adapter_name is not None and self.__llm_backend__ != 'pt':
      raise NotImplementedError(f'Adapter is not supported with {self.__llm_backend__}.')

    if isinstance(self.runner._runner_handle, DummyRunnerHandle):
      if os.getenv('BENTO_PATH') is not None:
        raise RuntimeError('Runner client failed to set up correctly.')
      else:
        self.runner.init_local(quiet=True)
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
      generator = self.runner.generate_iterator.async_stream(
        prompt_token_ids, request_id, stop=list(stop), adapter_name=adapter_name, **config.model_dump(flatten=True)
      )
      generator = bentoml.io.SSE.from_iterator(generator)
    except Exception as err:
      raise RuntimeError(f'Failed to start generation task: {err}') from err

    try:
      async for out in generator:
        out = out.data
        generated = GenerationOutput.from_runner(out).with_options(prompt=prompt)
        delta_outputs = [None] * len(generated.outputs)
        for output in generated.outputs:
          i = output.index
          delta_tokens, delta_text = output.token_ids[previous_num_tokens[i] :], output.text[len(previous_texts[i]) :]
          previous_texts[i], previous_num_tokens[i] = output.text, len(output.token_ids)
          delta_outputs[i] = output.with_options(text=delta_text, token_ids=delta_tokens)
        yield generated.with_options(outputs=delta_outputs)
    except Exception as err:
      raise RuntimeError(f'Exception caught during generation: {err}') from err

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

  __llm_dtype__: t.Union[LiteralDtype, t.Literal['auto', 'half', 'float']] = 'auto'
  __llm_torch_dtype__: 'torch.dtype' = None
  __llm_config__: t.Optional[LLMConfig] = None
  __llm_backend__: LiteralBackend = None
  __llm_quantization_config__: t.Optional[t.Union[transformers.BitsAndBytesConfig, transformers.GPTQConfig, transformers.AwqConfig]] = None
  __llm_runner__: t.Optional[Runner[M, T]] = None
  __llm_model__: t.Optional[M] = None
  __llm_tokenizer__: t.Optional[T] = None
  __llm_adapter_map__: t.Optional[ResolvedAdapterMap] = None
  __llm_trust_remote_code__: bool = False

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
    embedded=False,
    dtype='auto',
    low_cpu_mem_usage=True,
    max_model_len=None,
    gpu_memory_utilization=0.9,
    _eager=True,
    **attrs,
  ):
    torch_dtype = attrs.pop('torch_dtype', None)  # backward compatible
    if torch_dtype is not None:
      warnings.warn(
        'The argument "torch_dtype" is deprecated and will be removed in the future. Please use "dtype" instead.', DeprecationWarning, stacklevel=3
      )
      dtype = torch_dtype
    _local = False
    if validate_is_path(model_id):
      model_id, _local = resolve_filepath(model_id), True
    backend = getenv('backend', default=backend)
    if backend is None:
      backend = self._cascade_backend()
    dtype = getenv('dtype', default=dtype, var=['TORCH_DTYPE'])
    if dtype is None:
      logger.warning('Setting dtype to auto. Inferring from framework specific models')
      dtype = 'auto'
    quantize = getenv('quantize', default=quantize, var=['QUANITSE'])
    attrs.update({'low_cpu_mem_usage': low_cpu_mem_usage})
    # parsing tokenizer and model kwargs, as the hierarchy is param pass > default
    model_attrs, tokenizer_attrs = flatten_attrs(**attrs)
    if model_tag is None:
      model_tag, model_version = self._make_tag_components(model_id, model_version, backend=backend)
      if model_version:
        model_tag = f'{model_tag}:{model_version}'

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
      LLM__model_attrs=model_attrs,
      LLM__tokenizer_attrs=tokenizer_attrs,
      llm_dtype__=dtype.lower(),
      llm_backend__=backend,
      llm_config__=llm_config,
      llm_trust_remote_code__=trust_remote_code,
    )
    if _eager:
      try:
        model = bentoml.models.get(self.tag)
      except bentoml.exceptions.NotFound:
        model = openllm.serialisation.import_model(self, trust_remote_code=self.trust_remote_code)
      # resolve the tag
      self._tag = model.tag
    if not _eager and embedded:
      raise RuntimeError("Embedded mode is not supported when '_eager' is False.")
    if embedded:
      logger.warning('NOT RECOMMENDED in production and SHOULD ONLY used for development.')
      self.runner.init_local(quiet=True)

  class _Quantise:
    @staticmethod
    def pt(llm: LLM, quantise=None): return quantise
    @staticmethod
    def vllm(llm: LLM, quantise=None): return quantise
    @staticmethod
    def ctranslate(llm: LLM, quantise=None):
      if quantise in {'int4', 'awq', 'gptq', 'squeezellm'}: raise ValueError(f"Quantisation '{quantise}' is not supported for backend 'ctranslate'")
      if quantise == 'int8': quantise = 'int8_float16' if llm._has_gpus else 'int8_float32'
      return quantise

  @apply(lambda val: tuple(str.lower(i) if i else i for i in val))
  def _make_tag_components(self, model_id: str, model_version: str | None, backend: str) -> tuple[str, str | None]:
    model_id, *maybe_revision = model_id.rsplit(':')
    if len(maybe_revision) > 0:
      if model_version is not None:
        logger.warning("revision is specified (%s). 'model_version=%s' will be ignored.", maybe_revision[0], model_version)
      model_version = maybe_revision[0]
    if validate_is_path(model_id):
      model_id, model_version = resolve_filepath(model_id), first_not_none(model_version, default=generate_hash_from_file(model_id))
    return f'{backend}-{normalise_model_name(model_id)}', model_version

  @functools.cached_property
  def _has_gpus(self):
    try:
      from cuda import cuda

      err, *_ = cuda.cuInit(0)
      if err != cuda.CUresult.CUDA_SUCCESS: raise RuntimeError('Failed to initialise CUDA runtime binding.')
      err, _ = cuda.cuDeviceGetCount()
      if err != cuda.CUresult.CUDA_SUCCESS: raise RuntimeError('Failed to get CUDA device count.')
      return True
    except (ImportError, RuntimeError):
      return False

  @property
  def _torch_dtype(self):
    import torch, transformers

    _map = _torch_dtype_mapping()
    if not isinstance(self.__llm_torch_dtype__, torch.dtype):
      try:
        hf_config = transformers.AutoConfig.from_pretrained(self.bentomodel.path, trust_remote_code=self.trust_remote_code)
      except OpenLLMException:
        hf_config = transformers.AutoConfig.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
      config_dtype = getattr(hf_config, 'torch_dtype', None)
      if config_dtype is None:
        config_dtype = torch.float32
      if self.__llm_dtype__ == 'auto':
        if config_dtype == torch.float32:
          torch_dtype = torch.float16
        else:
          torch_dtype = config_dtype
      else:
        if self.__llm_dtype__ not in _map:
          raise ValueError(f"Unknown dtype '{self.__llm_dtype__}'")
        torch_dtype = _map[self.__llm_dtype__]
      self.__llm_torch_dtype__ = torch_dtype
    return self.__llm_torch_dtype__

  @property
  def _model_attrs(self):
    return {**self.import_kwargs[0], **self.__model_attrs}

  @_model_attrs.setter
  def _model_attrs(self, model_attrs):
    self.__model_attrs = model_attrs

  @property
  def _tokenizer_attrs(self):
    return {**self.import_kwargs[1], **self.__tokenizer_attrs}

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
      del self.__llm_model__, self.__llm_tokenizer__, self.__llm_adapter_map__
    except AttributeError:
      pass

  def __repr_args__(self):
    yield from (
      ('model_id', self._model_id if not self._local else self.tag.name),
      ('revision', self._revision if self._revision else self.tag.version),
      ('backend', self.__llm_backend__),
      ('type', self.llm_type),
    )

  def __repr__(self) -> str:
    return f'{self.__class__.__name__} {orjson.dumps({k: v for k, v in self.__repr_args__()}, option=orjson.OPT_INDENT_2).decode()}'

  @property
  def import_kwargs(self):
    return {'device_map': 'auto' if self._has_gpus else None, 'torch_dtype': self._torch_dtype}, {'padding_side': 'left', 'truncation_side': 'left'}

  @property
  def trust_remote_code(self):
    env = os.getenv('TRUST_REMOTE_CODE')
    if env is not None:
      return check_bool_env('TRUST_REMOTE_CODE', env)
    return self.__llm_trust_remote_code__

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
    return openllm.serialisation.get(self)

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
  def has_adapters(self):
    return self._adapter_map is not None

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
    from ._runners import runner

    if self.__llm_runner__ is None:
      self.__llm_runner__ = runner(self)
    return self.__llm_runner__

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
    logger.warning('`prepare_for_training` is deprecated and will be removed in the future. Use `prepare` instead.')
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
      model = openllm.serialisation.load_model(self, *self._model_decls, **self._model_attrs)
      # If OOM, then it is probably you don't have enough VRAM to run this model.
      if self.__llm_backend__ == 'pt':
        import torch

        loaded_in_kbit = (
          getattr(model, 'is_loaded_in_8bit', False) or getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_quantized', False)
        )
        if torch.cuda.is_available() and torch.cuda.device_count() == 1 and not loaded_in_kbit:
          try:
            model = model.to('cuda')
          except Exception as err:
            raise OpenLLMException(f'Failed to load model into GPU: {err}.\n') from err
        if self.has_adapters:
          logger.debug('Applying the following adapters: %s', self.adapter_map)
          for adapter_dict in self.adapter_map.values():
            for adapter_name, (peft_config, peft_model_id) in adapter_dict.items():
              model.load_adapter(peft_model_id, adapter_name, peft_config=peft_config)
      self.__llm_model__ = model
    return self.__llm_model__

  @property
  def config(self):
    import transformers

    if self.__llm_config__ is None:
      if self.__llm_backend__ == 'ctranslate':
        try:
          config = transformers.AutoConfig.from_pretrained(self.bentomodel.path_of('/hf'), trust_remote_code=self.trust_remote_code)
        except OpenLLMException:
          config = transformers.AutoConfig.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        for architecture in config.architectures:
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


@functools.lru_cache(maxsize=1)
def _torch_dtype_mapping() -> dict[str, torch.dtype]:
  import torch

  return {
    'half': torch.float16,
    'float16': torch.float16,  #
    'float': torch.float32,
    'float32': torch.float32,  #
    'bfloat16': torch.bfloat16,
  }


def normalise_model_name(name: str) -> str:
  return os.path.basename(resolve_filepath(name)) if validate_is_path(name) else inflection.dasherize(name.replace('/', '--'))


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
