# mypy: disable-error-code="name-defined,attr-defined"
from __future__ import annotations
import asyncio
import functools
import inspect
import logging
import os
import types
import typing as t
import warnings

import attr
import inflection
import orjson

from huggingface_hub import hf_hub_download

import bentoml
import openllm
import openllm_core

from bentoml._internal.models.model import ModelSignature
from bentoml._internal.runner.runner_handle import DummyRunnerHandle
from openllm_core._schema import GenerationOutput
from openllm_core._strategies import CascadingResourceStrategy
from openllm_core._typing_compat import AdaptersMapping
from openllm_core._typing_compat import AdaptersTuple
from openllm_core._typing_compat import AdapterType
from openllm_core._typing_compat import Concatenate
from openllm_core._typing_compat import DictStrAny
from openllm_core._typing_compat import LiteralBackend
from openllm_core._typing_compat import LiteralQuantise
from openllm_core._typing_compat import LiteralSerialisation
from openllm_core._typing_compat import LLMRunnable
from openllm_core._typing_compat import LLMRunner
from openllm_core._typing_compat import M
from openllm_core._typing_compat import ModelSignatureDict
from openllm_core._typing_compat import ParamSpec
from openllm_core._typing_compat import T
from openllm_core._typing_compat import TupleAny
from openllm_core.prompts import PromptTemplate
from openllm_core.utils import LazyLoader
from openllm_core.utils import ReprMixin
from openllm_core.utils import apply
from openllm_core.utils import codegen
from openllm_core.utils import converter
from openllm_core.utils import first_not_none
from openllm_core.utils import generate_hash_from_file
from openllm_core.utils import is_async_callable
from openllm_core.utils import is_peft_available
from openllm_core.utils import is_torch_available
from openllm_core.utils import is_vllm_available
from openllm_core.utils import normalize_attrs_to_model_tokenizer_pair
from openllm_core.utils import resolve_filepath
from openllm_core.utils import validate_is_path

from ._quantisation import infer_quantisation_config
from .exceptions import ForbiddenAttributeError
from .exceptions import OpenLLMException
from .serialisation.constants import PEFT_CONFIG_NAME

if t.TYPE_CHECKING:
  import peft
  import torch
  import transformers

  from openllm_core._configuration import LLMConfig
  from openllm_core.utils.representation import ReprArgs

else:
  transformers = LazyLoader('transformers', globals(), 'transformers')
  torch = LazyLoader('torch', globals(), 'torch')
  peft = LazyLoader('peft', globals(), 'peft')

ResolvedAdaptersMapping = t.Dict[AdapterType, t.Dict[str, t.Tuple['peft.PeftConfig', str]]]

P = ParamSpec('P')

logger = logging.getLogger(__name__)

def normalise_model_name(name: str) -> str:
  if validate_is_path(name): return os.path.basename(resolve_filepath(name))
  name = name.replace('/', '--')
  return inflection.dasherize(name)

def resolve_peft_config_type(adapter_map: dict[str, str | None]) -> AdaptersMapping:
  '''Resolve the type of the PeftConfig given the adapter_map.

  This is similar to how PeftConfig resolve its config type.

  Args:
  adapter_map: The given mapping from either SDK or CLI. See CLI docs for more information.
  '''
  resolved: AdaptersMapping = {}
  _has_set_default = False
  for path_or_adapter_id, name in adapter_map.items():
    resolve_name = name
    if resolve_name is None:
      if _has_set_default: raise ValueError('Only one adapter can be set as default.')
      resolve_name = 'default'
      _has_set_default = True
    if os.path.isfile(os.path.join(path_or_adapter_id, PEFT_CONFIG_NAME)):
      config_file = os.path.join(path_or_adapter_id, PEFT_CONFIG_NAME)
    else:
      try:
        config_file = hf_hub_download(path_or_adapter_id, PEFT_CONFIG_NAME)
      except Exception as err:
        raise ValueError(f"Can't find '{PEFT_CONFIG_NAME}' at '{path_or_adapter_id}'") from err
    with open(config_file, 'r') as file:
      resolved_config = orjson.loads(file.read())
    # all peft_type should be available in PEFT_CONFIG_NAME
    _peft_type: AdapterType = resolved_config['peft_type'].lower()
    if _peft_type not in resolved: resolved[_peft_type] = ()
    resolved[_peft_type] += (_AdaptersTuple((path_or_adapter_id, resolve_name, resolved_config)),)
  return resolved

_reserved_namespace = {'model', 'tokenizer', 'runner', 'import_kwargs'}
_AdaptersTuple: type[AdaptersTuple] = codegen.make_attr_tuple_class('AdaptersTuple', ['adapter_id', 'name', 'config'])
InferenceReturnType = t.Literal['text', 'object', 'sse']

@attr.define(slots=True, repr=False, init=False)
class LLM(t.Generic[M, T]):
  _model_id: str
  _revision: str | None
  quantization_config: transformers.BitsAndBytesConfig | transformers.GPTQConfig | transformers.AwqConfig | None
  _quantize: LiteralQuantise | None
  _model_decls: TupleAny
  _model_attrs: DictStrAny
  _tokenizer_attrs: DictStrAny
  _tag: bentoml.Tag
  _adapter_map: AdaptersMapping | None
  _serialisation: LiteralSerialisation
  _local: bool
  _prompt_template: PromptTemplate | None
  _system_message: str | None

  __llm_config__: LLMConfig | None = None
  __llm_backend__: LiteralBackend = None
  __llm_runner__: t.Optional[LLMRunner[M, T]] = None
  __llm_model__: t.Optional[M] = None
  __llm_tokenizer__: t.Optional[T] = None
  __llm_adapter_map__: t.Optional[ResolvedAdaptersMapping] = None

  device: 'torch.device' | None = None

  def __attrs_post_init__(self) -> None:
    if self.__llm_backend__ == 'pt': self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  def __init__(self,
               model_id: str,
               model_version: str | None = None,
               model_tag: str | bentoml.Tag | None = None,
               prompt_template: PromptTemplate | str | None = None,
               system_message: str | None = None,
               llm_config: LLMConfig | None = None,
               backend: LiteralBackend | None = None,
               *args: t.Any,
               quantize: LiteralQuantise | None = None,
               quantization_config: transformers.BitsAndBytesConfig | transformers.GPTQConfig | transformers.AwqConfig | None = None,
               adapter_map: dict[str, str | None] | None = None,
               serialisation: LiteralSerialisation = 'safetensors',
               **attrs: t.Any):
    # low_cpu_mem_usage is only available for model
    # this is helpful on system with low memory to avoid OOM
    low_cpu_mem_usage = attrs.pop('low_cpu_mem_usage', True)

    _local = False
    if validate_is_path(model_id): model_id, _local = resolve_filepath(model_id), True
    quantize = first_not_none(quantize, t.cast(t.Optional[LiteralQuantise], os.getenv('OPENLLM_QUANTIZE')), default=None)
    if backend is None: backend = 'vllm' if is_vllm_available() else 'pt'

    # quantization setup
    if quantization_config and quantize:
      logger.warning("Both 'quantization_config' and 'quantize' are specified. 'quantize' will be ignored.")
    elif quantization_config is None and quantize is not None:
      # in case users input `tokenizer` to __init__, default to the _model_id
      if quantize == 'gptq': attrs.setdefault('tokenizer', model_id)
      # TODO: support AWQConfig
      quantization_config, attrs = infer_quantisation_config(self, quantize, **attrs)

    attrs.update({'low_cpu_mem_usage': low_cpu_mem_usage, 'quantization_config': quantization_config, 'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32})
    model_kwds, tokenizer_kwds = self.import_kwargs

    # parsing tokenizer and model kwargs, as the hierarchy is param pass > default
    normalized_model_kwds, normalized_tokenizer_kwds = normalize_attrs_to_model_tokenizer_pair(**attrs)
    # NOTE: Save the args and kwargs for latter load
    model_attrs = {**model_kwds, **normalized_model_kwds}
    tokenizer_attrs = {**tokenizer_kwds, **normalized_tokenizer_kwds}

    if adapter_map is not None and not is_peft_available():
      raise RuntimeError("LoRA adapter requires 'peft' to be installed. Make sure to install OpenLLM with 'pip install \"openllm[fine-tune]\"'")
    if adapter_map: logger.debug('OpenLLM will apply the following adapters layers: %s', list(adapter_map))

    if isinstance(prompt_template, str): prompt_template = PromptTemplate(prompt_template)

    if model_tag is None:
      model_tag, model_version = self._make_tag_components(model_id, model_version, backend=backend)
      if model_version: model_tag = f'{model_tag}:{model_version}'

    self.__attrs_init__(model_id=model_id,
                        revision=model_version,
                        tag=bentoml.Tag.from_taglike(model_tag),
                        quantization_config=quantization_config,
                        quantize=quantize,
                        model_decls=args,
                        model_attrs=model_attrs,
                        tokenizer_attrs=tokenizer_attrs,
                        adapter_map=resolve_peft_config_type(adapter_map) if adapter_map is not None else None,
                        serialisation=serialisation,
                        local=_local,
                        prompt_template=prompt_template,
                        system_message=system_message,
                        llm_backend__=backend,
                        llm_config__=llm_config)

  @apply(lambda val: tuple(str.lower(i) if i else i for i in val))
  def _make_tag_components(self, model_id: str, model_version: str | None, backend: LiteralBackend) -> tuple[str, str]:
    '''Return a valid tag name (<backend>-<repo>--<model_id>) and its tag version.'''
    model_id, *maybe_revision = model_id.rsplit(':')
    if len(maybe_revision) > 0:
      if model_version is not None: logger.warning("revision is specified within 'model_id' (%s), and 'model_version=%s' will be ignored.", maybe_revision[0], model_version)
      model_version = maybe_revision[0]
    if validate_is_path(model_id): model_id, model_version = resolve_filepath(model_id), first_not_none(model_version, default=generate_hash_from_file(model_id))
    return f'{backend}-{normalise_model_name(model_id)}', model_version

  # yapf: disable
  def __setattr__(self,attr: str,value: t.Any)->None:
    if attr in _reserved_namespace:raise ForbiddenAttributeError(f'{attr} should not be set during runtime.')
    super().__setattr__(attr,value)
  @property
  def import_kwargs(self)->tuple[dict[str, t.Any],dict[str, t.Any]]: return {'device_map': 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None, 'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32}, {'padding_side': 'left', 'truncation_side': 'left'}
  @property
  def adapters_mapping(self)->AdaptersMapping|None:return self._adapter_map
  @property
  def trust_remote_code(self)->bool:return first_not_none(openllm_core.utils.check_bool_env('TRUST_REMOTE_CODE',False),default=self.config['trust_remote_code'])
  @property
  def runner_name(self)->str:return f"llm-{self.config['start_name']}-runner"
  @property
  def model_id(self)->str:return self._model_id
  @property
  def revision(self)->str:return t.cast(str, self._revision)
  @property
  def tag(self)->bentoml.Tag:return self._tag
  @property
  def bentomodel(self)->bentoml.Model:return openllm.serialisation.get(self)
  @property
  def config(self)->LLMConfig:
    if self.__llm_config__ is None: self.__llm_config__=openllm.AutoConfig.infer_class_from_llm(self).model_construct_env(**self._model_attrs)
    return self.__llm_config__
  def save_pretrained(self) -> bentoml.Model: return openllm.import_model(self.config['start_name'], model_id=self.model_id, model_version=self._revision, backend=self.__llm_backend__, quantize=self._quantize)
  # NOTE: The section below defines a loose contract with langchain's LLM interface.
  @property
  def llm_type(self)->str:return normalise_model_name(self._model_id)
  @property
  def identifying_params(self)->DictStrAny: return {'configuration': self.config.model_dump_json().decode(),'model_ids': orjson.dumps(self.config['model_ids']).decode(),'model_id': self.model_id}
  @property
  def llm_parameters(self)->tuple[tuple[tuple[t.Any,...],DictStrAny],DictStrAny]:return (self._model_decls,self._model_attrs),self._tokenizer_attrs
  def sanitize_parameters(self, prompt: str, **attrs: t.Any) -> tuple[str, DictStrAny, DictStrAny]: return self.config.sanitize_parameters(prompt,prompt_template=self._prompt_template,system_message=self._system_message,**attrs)
  def postprocess_generate(self, prompt: str, generation_result: t.Any, **attrs: t.Any) -> t.Any:
    if isinstance(generation_result, dict) and 'text' in generation_result: return generation_result['text']
    return self.config.postprocess_generate(prompt, generation_result, **attrs)
  # yapf: enable

  @property
  def model(self) -> M:
    if self.__llm_model__ is None:
      model = openllm.serialisation.load_model(self, *self._model_decls, **self._model_attrs)
      # If OOM, then it is probably you don't have enough VRAM to run this model.
      if self.__llm_backend__ == 'pt' and is_torch_available():
        loaded_in_kbit = getattr(model, 'is_loaded_in_8bit', False) or getattr(model, 'is_loaded_in_4bit', False) or getattr(model, 'is_quantized', False)
        if torch.cuda.is_available() and torch.cuda.device_count() == 1 and not loaded_in_kbit and not isinstance(model, transformers.Pipeline):
          try:
            model = model.to('cuda')
          except Exception as err:
            raise OpenLLMException(
                f'Failed to load {self} into GPU: {err}\nTip: If you run into OOM issue, maybe try different offload strategy. See https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/quantization#offload-between-cpu-and-gpu for more information.'
            ) from err
      self.__llm_model__ = model
    return self.__llm_model__

  @property
  def tokenizer(self) -> T:
    # NOTE: the signature of load_tokenizer here is the wrapper under _wrapped_load_tokenizer
    if self.__llm_tokenizer__ is None: self.__llm_tokenizer__ = openllm.serialisation.load_tokenizer(self, **self.llm_parameters[-1])
    return self.__llm_tokenizer__

  @property
  def runner(self) -> LLMRunner[M, T]:
    if self.__llm_runner__ is None: self.__llm_runner__ = _RunnerFactory(self)
    return self.__llm_runner__

  @staticmethod
  def inferece_wrapper(func: t.Callable[Concatenate[LLM[M, T], P], t.Any]) -> t.Callable[P, t.Any]:
    _is_async_gen = inspect.isasyncgenfunction(func)
    if not is_async_callable(func):
      if _is_async_gen: pass
      else: raise RuntimeError(f'assert_runner_is_ready can only be used on async function or async gen function, while {func} is not.')

    @functools.wraps(func)
    async def wrapped(self: LLM[M, T], *args: P.args, **kwargs: P.kwargs) -> t.Any:
      if isinstance(self.runner._runner_handle, DummyRunnerHandle):
        if os.getenv('BENTO_PATH') is not None: raise RuntimeError('Runner client failed to set up correctly.')
        else: self.runner.init_local(quiet=True)
      return await func(self, *args, **kwargs) if not _is_async_gen else func(self, *args, **kwargs)

    return wrapped

  @inferece_wrapper
  async def generate(self,
                     prompt: str,
                     stop: str | t.Iterable[str] | None = None,
                     stop_token_ids: list[int] | None = None,
                     return_type: InferenceReturnType = 'object',
                     **attrs: t.Any) -> GenerationOutput:
    prompt, *_ = self.sanitize_parameters(prompt, **attrs)
    config = self.config.model_construct_env(**attrs)

    if stop_token_ids is None: stop_token_ids = []
    if self.tokenizer.eos_token_id not in stop_token_ids: stop_token_ids.append(self.tokenizer.eos_token_id)
    if stop is None: stop = set()
    elif isinstance(stop, str): stop = {stop}
    else: stop = set(stop)
    for tid in stop_token_ids:
      if tid: stop.add(self.tokenizer.decode(tid))

    prompt_token_ids, request_id = self.tokenizer.encode(prompt), openllm_core.utils.gen_random_uuid()
    async for out in self.runner.generate.async_stream(prompt_token_ids, request_id, stop=stop, **config.model_dump()):
      pass

    if return_type == 'sse': return out
    generated = GenerationOutput.from_sse(out).with_options(prompt=prompt)
    if return_type == 'object': return generated
    elif return_type == 'text': return generated.outputs[0].text
    else: raise ValueError(f"'return_type' can only be one of ['object', 'text', 'sse'], while '{return_type}' is given.")

  @inferece_wrapper
  async def generate_iterator(self,
                              prompt: str,
                              stop: str | t.Iterable[str] | None = None,
                              stop_token_ids: list[int] | None = None,
                              return_type: InferenceReturnType = 'object',
                              **attrs: t.Any) -> t.AsyncGenerator[GenerationOutput | str, None]:
    prompt, *_ = self.sanitize_parameters(prompt, **attrs)
    config = self.config.model_construct_env(**attrs)

    if stop_token_ids is None: stop_token_ids = []
    if self.tokenizer.eos_token_id not in stop_token_ids: stop_token_ids.append(self.tokenizer.eos_token_id)
    if stop is None: stop = set()
    elif isinstance(stop, str): stop = {stop}
    else: stop = set(stop)
    for tid in stop_token_ids:
      if tid: stop.add(self.tokenizer.decode(tid))

    prompt_token_ids, request_id, pre = self.tokenizer.encode(prompt), openllm_core.utils.gen_random_uuid(), 0
    output_text = []
    async for out in self.runner.generate_iterator.async_stream(prompt_token_ids, request_id, stop=stop, **config.model_dump()):
      if return_type == 'sse': yield out
      elif return_type == 'object': yield GenerationOutput.from_sse(out).with_options(prompt=prompt)
      elif return_type == 'text':
        generated = GenerationOutput.from_sse(out).with_options(prompt=prompt)
        text_outputs = [output.text for output in generated.outputs]
        output_text = text_outputs[0].strip().split(' ')
        now = len(output_text) - 1
        if now > pre:
          yield ' '.join(output_text[pre:now]) + ' '
          pre = now
        if generated.finished: break
      else:
        raise ValueError(f"'return_type' can only be one of ['object', 'text', 'sse'], while '{return_type}' is given.")
    if return_type == 'text': yield ' '.join(output_text[pre:]) + ' '

def Runner(model_name: str,
           ensure_available: bool = False,
           init_local: bool = False,
           backend: LiteralBackend | None = None,
           llm_config: LLMConfig | None = None,
           **attrs: t.Any) -> LLMRunner[t.Any, t.Any]:
  '''Create a Runner for given LLM. For a list of currently supported LLM, check out 'openllm models'.

  > [!WARNING]
  > This method is now deprecated and in favor of 'openllm.LLM.runner'

  ```python
  runner = openllm.Runner("dolly-v2")

  @svc.on_startup
  def download():
    runner.download_model()
  ```

  if `init_local=True` (For development workflow), it will also enable `ensure_available`.
  Default value of `ensure_available` is None. If set then use that given value, otherwise fallback to the aforementioned behaviour.

  Args:
    model_name: Supported model name from 'openllm models'
    ensure_available: If True, it will download the model if it is not available. If False, it will skip downloading the model.
                      If False, make sure the model is available locally.
    backend: The given Runner implementation one choose for this Runner. If `OPENLLM_BACKEND` is set, it will respect it.
    llm_config: Optional ``openllm.LLMConfig`` to initialise this ``openllm.LLMRunner``.
    init_local: If True, it will initialize the model locally. This is useful if you want to run the model locally. (Symmetrical to bentoml.Runner.init_local())
    **attrs: The rest of kwargs will then be passed to the LLM. Refer to the LLM documentation for the kwargs behaviour
  '''
  if llm_config is None: llm_config = openllm.AutoConfig.for_model(model_name)
  model_id = attrs.get('model_id') or llm_config['env']['model_id_value']
  msg = f'''\
Using 'openllm.Runner' is now deprecated. Make sure to switch to the following syntax:

```python
llm = openllm.LLM('{model_id}')

svc = bentoml.Service('...', runners=[llm.runner])

@svc.api(...)
async def chat(input: str) -> str:
  await llm.generate(input)
```
  '''
  warnings.warn(msg, DeprecationWarning, stacklevel=3)
  attrs.update({
      'model_id': model_id,
      'quantize': llm_config['env']['quantize_value'],
      'serialisation': first_not_none(attrs.get('serialisation'), os.environ.get('OPENLLM_SERIALIZATION'), default=llm_config['serialisation']),
      'system_message': first_not_none(os.environ.get('OPENLLM_SYSTEM_MESSAGE'), attrs.get('system_message'), None),
      'prompt_template': first_not_none(os.environ.get('OPENLLM_PROMPT_TEMPLATE'), attrs.get('prompt_template'), None),
  })

  backend = t.cast(LiteralBackend, first_not_none(backend, default='vllm' if is_vllm_available() else 'pt'))
  if init_local: ensure_available = True
  llm = LLM(backend=backend, llm_config=llm_config, **attrs)
  if ensure_available: llm.save_pretrained()
  if init_local: llm.runner.init_local(quiet=True)
  return llm.runner

def _RunnerFactory(self: openllm.LLM[M, T],
                   /,
                   models: list[bentoml.Model] | None = None,
                   max_batch_size: int | None = None,
                   max_latency_ms: int | None = None,
                   scheduling_strategy: type[bentoml.Strategy] = CascadingResourceStrategy,
                   *,
                   backend: LiteralBackend | None = None) -> LLMRunner[M, T]:
  backend = t.cast(LiteralBackend, first_not_none(backend, os.environ.get('OPENLLM_BACKEND'), default=self.__llm_backend__))

  models = models if models is not None else []
  # if os.environ.get('BENTO_PATH') is None:
  #   # Hmm we should only add this if it is not in the container environment
  #   # BentoML sets BENTO_PATH so we can use this as switch logic here.
  #   try:
  #     models.append(self.bentomodel)
  #   except bentoml.exceptions.NotFound as err:
  #     raise RuntimeError(f'Failed to locate {self.bentomodel}:{err}') from err
  try:
    models.append(self.bentomodel)
  except bentoml.exceptions.NotFound as err:
    raise RuntimeError(f'Failed to locate {self.bentomodel}:{err}') from err

  if backend == 'vllm':
    from ._runners import vLLMRunnable as OpenLLMRunnable
  else:
    from ._runners import PyTorchRunnable as OpenLLMRunnable

  if self._prompt_template: prompt_template = self._prompt_template.to_string()
  elif hasattr(self.config, 'default_prompt_template'): prompt_template = self.config.default_prompt_template
  else: prompt_template = None

  if self._system_message: system_message = self._system_message
  elif hasattr(self.config, 'default_system_message'): system_message = self.config.default_system_message
  else: system_message = None

  def _wrapped_repr_keys(_: LLMRunner[M, T]) -> set[str]:
    return {'config', 'llm_type', 'runner_methods', 'backend', 'llm_tag'}

  def _generate_sync(_: LLMRunner[M, T], prompt: str, **kwargs: t.Any) -> t.Any:
    # NOTE: This should only be used with LangChain
    async def infer():
      return await self.generate(prompt, **kwargs)

    return asyncio.run(infer())

  def _wrapped_repr_args(_: LLMRunner[M, T]) -> ReprArgs:
    yield 'runner_methods', {method.name: {'batchable': method.config.batchable, 'batch_dim': method.config.batch_dim if method.config.batchable else None} for method in _.runner_methods}
    yield 'config', self.config.model_dump(flatten=True)
    yield 'llm_type', _.llm_type
    yield 'backend', backend
    yield 'llm_tag', self.tag

  return types.new_class(self.__class__.__name__ + 'Runner', (bentoml.Runner,),
                         exec_body=lambda ns: ns.update({
                             'llm_type': self.llm_type,
                             'identifying_params': self.identifying_params,
                             'llm_tag': self.tag,
                             'llm': self,
                             'config': self.config,
                             'backend': backend,
                             'download_model': self.save_pretrained,
                             '__module__': self.__module__,
                             '__call__': _generate_sync,
                             '__doc__': self.config['env'].start_docstring,
                             '__repr__': ReprMixin.__repr__,
                             '__repr_keys__': property(_wrapped_repr_keys),
                             '__repr_args__': _wrapped_repr_args,
                             'has_adapters': self.adapters_mapping is not None,
                             'prompt_template': prompt_template,
                             'system_message': system_message,
                         }))(OpenLLMRunnable,
                             name=self.runner_name,
                             embedded=False,
                             models=models,
                             max_batch_size=max_batch_size,
                             max_latency_ms=max_latency_ms,
                             scheduling_strategy=scheduling_strategy,
                             runnable_init_params=dict(llm=self),
                             method_configs=converter.unstructure({
                                 'generate': ModelSignature.from_dict(ModelSignatureDict(batchable=False)),
                                 'generate_iterator': ModelSignature.from_dict(ModelSignatureDict(batchable=False))
                             }))

# def llm_runnable_class(self: LLM[M, T], generate_sig: ModelSignature, generate_iterator_sig: ModelSignature) -> type[LLMRunnable[M, T]]:
#   class _Runnable(bentoml.Runnable):
#     SUPPORTED_RESOURCES = ('nvidia.com/gpu', 'amd.com/gpu', 'cpu')
#     SUPPORTS_CPU_MULTI_THREADING = True
#     backend = self.__llm_backend__
#
#     def __init__(__self: _Runnable):
#       # NOTE: The side effect of this line is that it will load the
#       # imported model during runner startup. So don't remove it!!
#       if not self.model: raise RuntimeError('Failed to load the model correctly (See traceback above)')
#       if self.adapters_mapping is not None:
#         logger.info('Applying LoRA to %s...', self.runner_name)
#         self.apply_adapter(inference_mode=True, load_adapters='all')
#
#     def set_adapter(__self: _Runnable, adapter_name: str) -> None:
#       if self.__llm_adapter_map__ is None: raise ValueError('No adapters available for current running server.')
#       elif not isinstance(self.model, peft.PeftModel): raise RuntimeError('Model is not a PeftModel')
#       if adapter_name != 'default': self.model.set_adapter(adapter_name)
#       logger.info('Successfully apply LoRA layer %s', adapter_name)
#
#     @bentoml.Runnable.method(**method_signature(generate_sig))  # type: ignore
#     def __call__(__self: _Runnable, prompt: str, **attrs: t.Any) -> list[t.Any]:
#       prompt, *_ = self.sanitize_parameters(prompt, **attrs)
#       adapter_name = attrs.pop('adapter_name', None)
#       if adapter_name is not None: __self.set_adapter(adapter_name)
#       return self.generate(prompt, **attrs)
#
#     @bentoml.Runnable.method(**method_signature(generate_sig))  # type: ignore
#     def generate(__self: _Runnable, prompt: str, **attrs: t.Any) -> list[t.Any]:
#       prompt, *_ = self.sanitize_parameters(prompt, **attrs)
#       adapter_name = attrs.pop('adapter_name', None)
#       if adapter_name is not None: __self.set_adapter(adapter_name)
#       return self.generate(prompt, **attrs)
#
#     @bentoml.Runnable.method(**method_signature(generate_sig))  # type: ignore
#     def generate_one(__self: _Runnable, prompt: str, stop: list[str], **attrs: t.Any) -> t.Sequence[dict[t.Literal['generated_text'], str]]:
#       prompt, *_ = self.sanitize_parameters(prompt, **attrs)
#       adapter_name = attrs.pop('adapter_name', None)
#       if adapter_name is not None: __self.set_adapter(adapter_name)
#       return self.generate_one(prompt, stop, **attrs)
#
#     @bentoml.Runnable.method(**method_signature(generate_iterator_sig))
#     def generate_iterator(__self: _Runnable, prompt: str, **attrs: t.Any) -> t.Generator[str, None, str]:
#       prompt, *_ = self.sanitize_parameters(prompt, **attrs)
#       adapter_name = attrs.pop('adapter_name', None)
#       if adapter_name is not None: __self.set_adapter(adapter_name)
#       pre = 0
#       for outputs in self.generate_iterator(prompt, request_id=openllm_core.utils.gen_random_uuid(), **attrs):
#         output_text = outputs['text'].strip().split(' ')
#         now = len(output_text) - 1
#         if now > pre:
#           yield ' '.join(output_text[pre:now]) + ' '
#           pre = now
#       yield ' '.join(output_text[pre:]) + ' '
#       return ' '.join(output_text) + ' '
#
#     @bentoml.Runnable.method(**method_signature(generate_sig))  # type: ignore
#     async def vllm_generate(__self: _Runnable, prompt: str, **attrs: t.Any) -> t.AsyncGenerator[list[t.Any], None]:
#       # TODO: PEFT support
#       attrs.pop('adapter_name', None)
#
#       stop: str | t.Iterable[str] | None = attrs.pop('stop', None)
#       # echo = attrs.pop('echo', False)
#       stop_token_ids: list[int] | None = attrs.pop('stop_token_ids', None)
#       temperature = attrs.pop('temperature', self.config['temperature'])
#       top_p = attrs.pop('top_p', self.config['top_p'])
#       request_id: str | None = attrs.pop('request_id', None)
#       if request_id is None: raise ValueError('request_id must not be None.')
#       prompt, *_ = self.sanitize_parameters(prompt, **attrs)
#       if openllm_core.utils.DEBUG: logger.debug('Prompt:\n%s', prompt)
#
#       if stop_token_ids is None: stop_token_ids = []
#       stop_token_ids.append(self.tokenizer.eos_token_id)
#       stop_: set[str] = set()
#       if isinstance(stop, str) and stop != '': stop_.add(stop)
#       elif isinstance(stop, list) and stop != []: stop_.update(stop)
#       for tid in stop_token_ids:
#         if tid: stop_.add(self.tokenizer.decode(tid))
#
#       if temperature <= 1e-5: top_p = 1.0
#       config = self.config.model_construct_env(stop=list(stop_), top_p=top_p, **attrs)
#       sampling_params = config.to_sampling_config()
#
#       final_output = None
#       async for request_output in t.cast('vllm.AsyncLLMEngine', self.model).generate(prompt=prompt, sampling_params=sampling_params, request_id=request_id):
#         final_output = request_output
#       if final_output is None: raise ValueError("'output' should not be None")
#       # prompt = final_output.prompt
#       # if echo: text_outputs = [prompt + output.text for output in final_output.outputs]
#       # else: text_outputs = [output.text for output in final_output.outputs]
#       yield final_output
#
#     @bentoml.Runnable.method(**method_signature(generate_iterator_sig))
#     async def vllm_generate_iterator(__self: _Runnable, prompt: str, **attrs: t.Any) -> t.AsyncGenerator[str, None]:
#       # TODO: PEFT support
#       attrs.pop('adapter_name', None)
#
#       # pre = 0
#       # echo = attrs.pop('echo', False)
#       request_id: str | None = attrs.pop('request_id', None)
#       if request_id is None: raise ValueError('request_id must not be None.')
#
#       stop: str | t.Iterable[str] | None = attrs.pop('stop', None)
#       temperature = attrs.pop('temperature', self.config['temperature'])
#       top_p = attrs.pop('top_p', self.config['top_p'])
#       prompt, *_ = self.sanitize_parameters(prompt, **attrs)
#       if openllm_core.utils.DEBUG: logger.debug('Prompt:\n%s', repr(prompt))
#
#       stop_token_ids: list[int] | None = attrs.pop('stop_token_ids', None)
#       if stop_token_ids is None: stop_token_ids = []
#       stop_token_ids.append(self.tokenizer.eos_token_id)
#       stop_: set[str] = set()
#       if isinstance(stop, str) and stop != '': stop_.add(stop)
#       elif isinstance(stop, list) and stop != []: stop_.update(stop)
#       for tid in stop_token_ids:
#         if tid: stop_.add(self.tokenizer.decode(tid))
#
#       if temperature <= 1e-5: top_p = 1.0
#       config = self.config.model_construct_env(stop=list(stop_), temperature=temperature, top_p=top_p, **attrs)
#       async for request_output in t.cast('vllm.AsyncLLMEngine', self.model).generate(prompt=prompt, sampling_params=config.to_sampling_config(), request_id=request_id):
#         # yield request_output
#         if echo: text_outputs = [prompt + output.text for output in request_output.outputs]
#         else: text_outputs = [output.text for output in request_output.outputs]
#         output_text = text_outputs[0]
#         output_text = output_text.strip().split(' ')
#         now = len(output_text) - 1
#         if now > pre:
#           yield ' '.join(output_text[pre:now]) + ' '
#           pre = now
#       yield ' '.join(output_text[pre:]) + ' '
#
#   return types.new_class(self.__class__.__name__ + 'Runnable', (_Runnable,), {}, lambda ns: ns.update({
#       'SUPPORTED_RESOURCES': ('nvidia.com/gpu', 'amd.com/gpu', 'cpu'),
#       '__module__': self.__module__,
#       '__doc__': self.config['env'].start_docstring
#   }))
#
# def llm_runner_class(self: LLM[M, T]) -> type[LLMRunner[M, T]]:
#   def _wrapped_generate_run(__self: LLMRunner[M, T], prompt: str, **kwargs: t.Any) -> t.Any:
#     '''Wrapper for runner.generate.run() to handle the prompt and postprocessing.
#
#     This will be used for LangChain API.
#
#     Usage:
#
#     ```python
#     runner = openllm.Runner("dolly-v2", init_local=True)
#     runner("What is the meaning of life?")
#     ```
#     '''
#     prompt, generate_kwargs, postprocess_kwargs = self.sanitize_parameters(prompt, **kwargs)
#     return self.postprocess_generate(prompt, __self.generate.run(prompt, **generate_kwargs), **postprocess_kwargs)
#
#   def _wrapped_repr_keys(_: LLMRunner[M, T]) -> set[str]:
#     return {'config', 'llm_type', 'runner_methods', 'backend', 'llm_tag'}
#
#   def _wrapped_repr_args(__self: LLMRunner[M, T]) -> ReprArgs:
#     yield 'runner_methods', {method.name: {'batchable': method.config.batchable, 'batch_dim': method.config.batch_dim if method.config.batchable else None} for method in __self.runner_methods}
#     yield 'config', self.config.model_dump(flatten=True)
#     yield 'llm_type', __self.llm_type
#     yield 'backend', self.__llm_backend__
#     yield 'llm_tag', self.tag
#
#   if self._prompt_template: prompt_template = self._prompt_template.to_string()
#   elif hasattr(self.config, 'default_prompt_template'): prompt_template = self.config.default_prompt_template
#   else: prompt_template = None
#
#   if self._system_message: system_message = self._system_message
#   elif hasattr(self.config, 'default_system_message'): system_message = self.config.default_system_message
#   else: system_message = None
#
#   return types.new_class(self.__class__.__name__ + 'Runner', (bentoml.Runner,),
#                          exec_body=lambda ns: ns.update({
#                              'llm_type': self.llm_type,
#                              'identifying_params': self.identifying_params,
#                              'llm_tag': self.tag,
#                              'llm': self,
#                              'config': self.config,
#                              'backend': self.__llm_backend__,
#                              'download_model': self.save_pretrained,
#                              '__call__': _wrapped_generate_run,
#                              '__module__': self.__module__,
#                              '__doc__': self.config['env'].start_docstring,
#                              '__repr__': ReprMixin.__repr__,
#                              '__repr_keys__': property(_wrapped_repr_keys),
#                              '__repr_args__': _wrapped_repr_args,
#                              'has_adapters': self._adapter_map is not None,
#                              'prompt_template': prompt_template,
#                              'system_message': system_message,
#                          }))

__all__ = ['LLMRunner', 'LLMRunnable', 'Runner', 'LLM']
