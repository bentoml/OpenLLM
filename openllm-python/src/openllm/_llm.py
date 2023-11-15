# mypy: disable-error-code="name-defined,attr-defined"
from __future__ import annotations
import abc
import functools
import logging
import os
import types
import typing as t

import attr
import inflection
import orjson
from huggingface_hub import hf_hub_download

import bentoml
import openllm
import openllm_core
from bentoml._internal.models.model import ModelSignature
from bentoml._internal.runner.runner_handle import DummyRunnerHandle
from openllm_core._schemas import CompletionChunk, GenerationOutput
from openllm_core._typing_compat import (
  AdapterMap,
  AdapterTuple,
  AdapterType,
  DictStrAny,
  LiteralBackend,
  LiteralDtype,
  LiteralQuantise,
  LiteralSerialisation,
  M,
  ParamSpec,
  T,
  TupleAny,
)
from openllm_core.exceptions import MissingDependencyError
from openllm_core.prompts import PromptTemplate
from openllm_core.utils import (
  DEBUG,
  ENV_VARS_TRUE_VALUES,
  ReprMixin,
  apply,
  codegen,
  converter,
  first_not_none,
  flatten_attrs,
  generate_hash_from_file,
  get_debug_mode,
  get_disable_warnings,
  get_quiet_mode,
  is_peft_available,
  resolve_filepath,
  validate_is_path,
)

from ._quantisation import infer_quantisation_config
from ._strategies import CascadingResourceStrategy
from .exceptions import ForbiddenAttributeError, OpenLLMException
from .serialisation.constants import PEFT_CONFIG_NAME

if t.TYPE_CHECKING:
  import torch
  import transformers
  from peft.config import PeftConfig
  from peft.peft_model import PeftModel, PeftModelForCausalLM, PeftModelForSeq2SeqLM

  from bentoml._internal.runner.runnable import RunnableMethod
  from bentoml._internal.runner.runner import RunnerMethod
  from bentoml._internal.runner.runner_handle import RunnerHandle
  from bentoml._internal.runner.strategy import Strategy
  from openllm_core._configuration import LLMConfig
  from openllm_core.utils.representation import ReprArgs

ResolvedAdapterMap = t.Dict[AdapterType, t.Dict[str, t.Tuple['PeftConfig', str]]]

P = ParamSpec('P')

logger = logging.getLogger(__name__)


def normalise_model_name(name: str) -> str:
  if validate_is_path(name):
    return os.path.basename(resolve_filepath(name))
  name = name.replace('/', '--')
  return inflection.dasherize(name)


def resolve_peft_config_type(adapter_map: dict[str, str]) -> AdapterMap:
  """Resolve the type of the PeftConfig given the adapter_map.

  This is similar to how PeftConfig resolve its config type.

  Args:
  adapter_map: The given mapping from either SDK or CLI. See CLI docs for more information.
  """
  resolved: AdapterMap = {}
  _has_set_default = False
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
    # all peft_type should be available in PEFT_CONFIG_NAME
    _peft_type: AdapterType = resolved_config['peft_type'].lower()
    if _peft_type not in resolved:
      resolved[_peft_type] = ()
    resolved[_peft_type] += (_AdapterTuple((path_or_adapter_id, name, resolved_config)),)
  return resolved


_reserved_namespace = {'model', 'tokenizer', 'runner', 'import_kwargs'}
_AdapterTuple: type[AdapterTuple] = codegen.make_attr_tuple_class('AdapterTuple', ['adapter_id', 'name', 'config'])


@functools.lru_cache(maxsize=1)
def _torch_dtype_mapping():
  import torch

  return {
    'half': torch.float16,
    'float16': torch.float16,
    'float': torch.float32,
    'float32': torch.float32,
    'bfloat16': torch.bfloat16,
  }


@attr.define(slots=True, repr=False, init=False)
class LLM(t.Generic[M, T], ReprMixin):
  _model_id: str
  _revision: str | None
  _quantization_config: transformers.BitsAndBytesConfig | transformers.GPTQConfig | transformers.AwqConfig | None
  _quantise: LiteralQuantise | None
  _model_decls: TupleAny
  __model_attrs: DictStrAny
  __tokenizer_attrs: DictStrAny
  _tag: bentoml.Tag
  _adapter_map: AdapterMap | None
  _serialisation: LiteralSerialisation
  _local: bool
  _prompt_template: PromptTemplate | None
  _system_message: str | None

  __llm_torch_dtype__: LiteralDtype | t.Literal['auto', 'half', 'float'] = 'auto'
  __llm_config__: LLMConfig | None = None
  __llm_backend__: LiteralBackend = None  # type: ignore
  __llm_quantization_config__: transformers.BitsAndBytesConfig | transformers.GPTQConfig | transformers.AwqConfig | None = None
  __llm_runner__: t.Optional[LLMRunner[M, T]] = None
  __llm_model__: t.Optional[M] = None
  __llm_tokenizer__: t.Optional[T] = None
  __llm_adapter_map__: t.Optional[ResolvedAdapterMap] = None
  __llm_trust_remote_code__: bool = False

  def __init__(
    self,
    model_id: str,
    model_version: str | None = None,
    model_tag: str | bentoml.Tag | None = None,
    prompt_template: PromptTemplate | str | None = None,
    system_message: str | None = None,
    llm_config: LLMConfig | None = None,
    backend: LiteralBackend | None = None,
    *args: t.Any,
    quantize: LiteralQuantise | None = None,
    quantization_config: transformers.BitsAndBytesConfig
    | transformers.GPTQConfig
    | transformers.AwqConfig
    | None = None,
    adapter_map: dict[str, str] | None = None,
    serialisation: LiteralSerialisation = 'safetensors',
    trust_remote_code: bool = False,
    embedded: bool = False,
    torch_dtype: LiteralDtype | t.Literal['auto', 'half', 'float'] = 'auto',
    **attrs: t.Any,
  ):
    # low_cpu_mem_usage is only available for model this is helpful on system with low memory to avoid OOM
    low_cpu_mem_usage = attrs.pop('low_cpu_mem_usage', True)
    _local = False
    if validate_is_path(model_id):
      model_id, _local = resolve_filepath(model_id), True

    backend = first_not_none(
      backend, os.getenv('OPENLLM_BACKEND'), default='vllm' if openllm.utils.is_vllm_available() else 'pt'
    )
    torch_dtype = first_not_none(os.getenv('TORCH_DTYPE'), torch_dtype, default='auto')
    quantize = first_not_none(quantize, os.getenv('OPENLLM_QUANTIZE'), default=None)
    # elif quantization_config is None and quantize is not None:
    #   quantization_config, attrs = infer_quantisation_config(self, quantize, **attrs)
    attrs.update({'low_cpu_mem_usage': low_cpu_mem_usage})

    # parsing tokenizer and model kwargs, as the hierarchy is param pass > default
    model_attrs, tokenizer_attrs = flatten_attrs(**attrs)

    if adapter_map is not None and not is_peft_available():
      raise RuntimeError(
        "LoRA adapter requires 'peft' to be installed. Make sure to do 'pip install \"openllm[fine-tune]\"'"
      )
    if isinstance(prompt_template, str):
      prompt_template = PromptTemplate(prompt_template)
    if model_tag is None:
      model_tag, model_version = self._make_tag_components(model_id, model_version, backend=backend)
      if model_version:
        model_tag = f'{model_tag}:{model_version}'

    self.__attrs_init__(
      model_id=model_id,
      revision=model_version,
      tag=bentoml.Tag.from_taglike(model_tag),
      quantization_config=quantization_config,
      quantise=quantize,
      model_decls=args,
      adapter_map=resolve_peft_config_type(adapter_map) if adapter_map is not None else None,
      serialisation=serialisation,
      local=_local,
      prompt_template=prompt_template,
      system_message=system_message,
      LLM__model_attrs=model_attrs,
      LLM__tokenizer_attrs=tokenizer_attrs,
      llm_torch_dtype__=torch_dtype.lower(),
      llm_backend__=backend,
      llm_config__=llm_config,
      llm_trust_remote_code__=trust_remote_code,
    )

    try:
      model = bentoml.models.get(self.tag)
    except bentoml.exceptions.NotFound:
      model = openllm.serialisation.import_model(self, trust_remote_code=self.trust_remote_code)
    # resolve the tag
    self._tag = model.tag

    if embedded and not get_disable_warnings() and not get_quiet_mode():
      logger.warning(
        'You are using embedded mode, which means the models will be loaded into memory. This is often not recommended in production and should only be used for local development only.'
      )
      if not get_debug_mode():
        logger.info("To disable this warning, set 'OPENLLM_DISABLE_WARNING=True'")
      self.runner.init_local(quiet=True)

  @property
  def _torch_dtype(self) -> torch.dtype:
    import torch
    import transformers

    if not isinstance(self.__llm_torch_dtype__, torch.dtype):
      try:
        hf_config = transformers.AutoConfig.from_pretrained(
          self.bentomodel.path, trust_remote_code=self.trust_remote_code
        )
      except OpenLLMException:
        hf_config = transformers.AutoConfig.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
      config_dtype = getattr(hf_config, 'torch_dtype', None)
      if config_dtype is None:
        config_dtype = torch.float32
      if not torch.cuda.is_available():
        if self.__llm_torch_dtype__ in {'auto', 'half'}:
          logger.warning('"auto" and "half" are not supported on CPU. OpenLLM will default fallback to "float32".')
        torch_dtype = torch.float32  # we need to cast back to full precision if cuda is not available
      elif self.__llm_torch_dtype__ == 'auto':
        if config_dtype == torch.float32:
          torch_dtype = torch.float16  # following common practice
        else:
          torch_dtype = config_dtype
      else:
        if self.__llm_torch_dtype__ not in _torch_dtype_mapping():
          raise ValueError(f"Unknown dtype '{self.__llm_torch_dtype__}'")
        torch_dtype = _torch_dtype_mapping()[self.__llm_torch_dtype__]
      self.__llm_torch_dtype__ = torch_dtype
    return self.__llm_torch_dtype__

  @apply(lambda val: tuple(str.lower(i) if i else i for i in val))
  def _make_tag_components(self, model_id, model_version, backend) -> tuple[str, str | None]:
    model_id, *maybe_revision = model_id.rsplit(':')
    if len(maybe_revision) > 0:
      if model_version is not None:
        logger.warning(
          "revision is specified within 'model_id' (%s), and 'model_version=%s' will be ignored.",
          maybe_revision[0],
          model_version,
        )
      model_version = maybe_revision[0]
    if validate_is_path(model_id):
      model_id, model_version = (
        resolve_filepath(model_id),
        first_not_none(model_version, default=generate_hash_from_file(model_id)),
      )
    return f'{backend}-{normalise_model_name(model_id)}', model_version

  def __setattr__(self, attr, value):
    if attr in _reserved_namespace:
      raise ForbiddenAttributeError(f'{attr} should not be set during runtime.')
    super().__setattr__(attr, value)

  @property
  def _model_attrs(self) -> dict[str, t.Any]:
    return {**self.import_kwargs[0], **self.__model_attrs}

  @property
  def _tokenizer_attrs(self) -> dict[str, t.Any]:
    return {**self.import_kwargs[1], **self.__tokenizer_attrs}

  @property
  def __repr_keys__(self):
    return {'model_id', 'revision', 'backend', 'type'}

  def __repr_args__(self):
    yield 'model_id', self._model_id if not self._local else self.tag.name
    yield 'revision', self._revision if self._revision else self.tag.version
    yield 'backend', self.__llm_backend__
    yield 'type', self.llm_type

  @property
  def import_kwargs(self) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
    import torch

    return {'device_map': 'auto' if torch.cuda.is_available() else None, 'torch_dtype': self._torch_dtype}, {
      'padding_side': 'left',
      'truncation_side': 'left',
    }

  @property
  def trust_remote_code(self) -> bool:
    env = os.getenv('TRUST_REMOTE_CODE')
    if env is not None:
      return str(env).upper() in ENV_VARS_TRUE_VALUES
    return self.__llm_trust_remote_code__

  @property
  def runner_name(self) -> str:
    return f"llm-{self.config['start_name']}-runner"

  @property
  def model_id(self) -> str:
    return self._model_id

  @property
  def revision(self) -> str:
    return t.cast(str, self._revision)

  @property
  def tag(self) -> bentoml.Tag:
    return self._tag

  @property
  def bentomodel(self) -> bentoml.Model:
    return openllm.serialisation.get(self)

  @property
  def quantization_config(self) -> transformers.BitsAndBytesConfig | transformers.GPTQConfig | transformers.AwqConfig:
    if self.__llm_quantization_config__ is None:
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
  def has_adapters(self) -> bool:
    return self._adapter_map is not None

  @property
  def local(self) -> bool:
    return self._local

  @property
  def quantise(self) -> LiteralQuantise | None:
    return self._quantise

  # NOTE: The section below defines a loose contract with langchain's LLM interface.
  @property
  def llm_type(self) -> str:
    return normalise_model_name(self._model_id)

  @property
  def identifying_params(self) -> DictStrAny:
    return {
      'configuration': self.config.model_dump_json().decode(),
      'model_ids': orjson.dumps(self.config['model_ids']).decode(),
      'model_id': self.model_id,
    }

  @property
  def llm_parameters(self) -> tuple[tuple[tuple[t.Any, ...], DictStrAny], DictStrAny]:
    return (self._model_decls, self._model_attrs), self._tokenizer_attrs

  # NOTE: This section is the actual model, tokenizer, and config reference here.
  @property
  def config(self) -> LLMConfig:
    if self.__llm_config__ is None:
      self.__llm_config__ = openllm.AutoConfig.infer_class_from_llm(self).model_construct_env(**self._model_attrs)
    return self.__llm_config__

  @property
  def tokenizer(self) -> T:
    if self.__llm_tokenizer__ is None:
      self.__llm_tokenizer__ = openllm.serialisation.load_tokenizer(self, **self.llm_parameters[-1])
    return self.__llm_tokenizer__

  @property
  def runner(self) -> LLMRunner[M, T]:
    if self.__llm_runner__ is None:
      self.__llm_runner__ = _RunnerFactory(self)
    return self.__llm_runner__

  @property
  def model(self) -> M:
    if self.__llm_model__ is None:
      model = openllm.serialisation.load_model(self, *self._model_decls, **self._model_attrs)
      # If OOM, then it is probably you don't have enough VRAM to run this model.
      if self.__llm_backend__ == 'pt':
        import torch

        loaded_in_kbit = (
          getattr(model, 'is_loaded_in_8bit', False)
          or getattr(model, 'is_loaded_in_4bit', False)
          or getattr(model, 'is_quantized', False)
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
  def adapter_map(self) -> ResolvedAdapterMap:
    try:
      import peft as _  # noqa: F401
    except ImportError as err:
      raise MissingDependencyError(
        "Failed to import 'peft'. Make sure to do 'pip install \"openllm[fine-tune]\"'"
      ) from err
    if not self.has_adapters:
      raise AttributeError('Adapter map is not available.')
    assert self._adapter_map is not None
    if self.__llm_adapter_map__ is None:
      _map: ResolvedAdapterMap = {k: {} for k in self._adapter_map}
      for adapter_type, adapter_tuple in self._adapter_map.items():
        base = first_not_none(
          self.config['fine_tune_strategies'].get(adapter_type),
          default=self.config.make_fine_tune_config(adapter_type),
        )
        for adapter in adapter_tuple:
          _map[adapter_type][adapter.name] = (base.with_config(**adapter.config).build(), adapter.adapter_id)
      self.__llm_adapter_map__ = _map
    return self.__llm_adapter_map__

  def prepare_for_training(
    self, adapter_type: AdapterType = 'lora', use_gradient_checking: bool = True, **attrs: t.Any
  ) -> tuple[PeftModel | PeftModelForCausalLM | PeftModelForSeq2SeqLM, T]:
    from peft.mapping import get_peft_model
    from peft.utils.other import prepare_model_for_kbit_training

    peft_config = (
      self.config['fine_tune_strategies']
      .get(adapter_type, self.config.make_fine_tune_config(adapter_type))
      .train()
      .with_config(**attrs)
      .build()
    )
    if self.has_adapters:
      raise ValueError('Adapter should not be specified when fine-tuning.')
    model = get_peft_model(
      prepare_model_for_kbit_training(self.model, use_gradient_checkpointing=use_gradient_checking), peft_config
    )
    if DEBUG:
      model.print_trainable_parameters()  # type: ignore[no-untyped-call]
    return model, self.tokenizer

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
    config = self.config.model_construct_env(**attrs)
    texts: list[list[str]] = [[]] * config['n']
    token_ids: list[list[int]] = [[]] * config['n']
    final_result: GenerationOutput | None = None
    async for result in self.generate_iterator(
      prompt, prompt_token_ids, stop, stop_token_ids, request_id, adapter_name, **config.model_dump(flatten=True)
    ):
      for output in result.outputs:
        texts[output.index].append(output.text)
        token_ids[output.index].extend(output.token_ids)
      final_result = result
    if final_result is None:
      raise RuntimeError('No result is returned.')
    return final_result.with_options(
      prompt=prompt,
      outputs=[
        output.with_options(text=''.join(texts[output.index]), token_ids=token_ids[output.index])
        for output in final_result.outputs
      ],
    )

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
    if isinstance(self.runner._runner_handle, DummyRunnerHandle):
      if os.getenv('BENTO_PATH') is not None:
        raise RuntimeError('Runner client failed to set up correctly.')
      else:
        self.runner.init_local(quiet=True)

    config = self.config.model_construct_env(**attrs)

    if stop_token_ids is None:
      stop_token_ids = []
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

    if request_id is None:
      request_id = openllm_core.utils.gen_random_uuid()
    previous_texts, previous_num_tokens = [''] * config['n'], [0] * config['n']
    async for out in self.runner.generate_iterator.async_stream(
      prompt_token_ids, request_id, stop, adapter_name, **config.model_dump(flatten=True)
    ):
      generated = GenerationOutput.from_runner(out).with_options(prompt=prompt)
      delta_outputs = t.cast(t.List[CompletionChunk], [None] * len(generated.outputs))
      if generated.finished:
        break
      for output in generated.outputs:
        i = output.index
        delta_tokens, delta_text = output.token_ids[previous_num_tokens[i] :], output.text[len(previous_texts[i]) :]
        previous_texts[i], previous_num_tokens[i] = output.text, len(output.token_ids)
        delta_outputs[i] = output.with_options(text=delta_text, token_ids=delta_tokens)
      yield generated.with_options(outputs=delta_outputs)


def _RunnerFactory(
  self: openllm.LLM[M, T],
  /,
  models: list[bentoml.Model] | None = None,
  max_batch_size: int | None = None,
  max_latency_ms: int | None = None,
  scheduling_strategy: type[bentoml.Strategy] = CascadingResourceStrategy,
  *,
  backend: LiteralBackend | None = None,
) -> LLMRunner[M, T]:
  from ._runners import runnable

  backend = t.cast(
    LiteralBackend, first_not_none(backend, os.environ.get('OPENLLM_BACKEND'), default=self.__llm_backend__)
  )

  models = models if models is not None else []
  try:
    models.append(self.bentomodel)
  except bentoml.exceptions.NotFound as err:
    raise RuntimeError(f'Failed to locate {self.bentomodel}:{err}') from err

  if self._prompt_template:
    prompt_template = self._prompt_template.to_string()
  elif hasattr(self.config, 'default_prompt_template'):
    prompt_template = self.config.default_prompt_template
  else:
    prompt_template = None
  if self._system_message:
    system_message = self._system_message
  elif hasattr(self.config, 'default_system_message'):
    system_message = self.config.default_system_message
  else:
    system_message = None

  def _wrapped_repr_keys(_: LLMRunner[M, T]) -> set[str]:
    return {'config', 'llm_type', 'runner_methods', 'backend', 'llm_tag'}

  def _wrapped_repr_args(_: LLMRunner[M, T]) -> ReprArgs:
    yield (
      'runner_methods',
      {
        method.name: {
          'batchable': method.config.batchable,
          'batch_dim': method.config.batch_dim if method.config.batchable else None,
        }
        for method in _.runner_methods
      },
    )
    yield 'config', self.config.model_dump(flatten=True)
    yield 'llm_type', self.llm_type
    yield 'backend', backend
    yield 'llm_tag', self.tag

  return types.new_class(
    self.__class__.__name__ + 'Runner',
    (bentoml.Runner,),
    exec_body=lambda ns: ns.update(
      {
        'llm_type': self.llm_type,
        'identifying_params': self.identifying_params,
        'llm_tag': self.tag,
        'llm': self,
        'config': self.config,
        'backend': backend,
        '__module__': self.__module__,
        '__doc__': getattr(openllm_core.config, f'START_{self.config["model_name"].upper()}_COMMAND_DOCSTRING'),
        '__repr__': ReprMixin.__repr__,
        '__repr_keys__': property(_wrapped_repr_keys),
        '__repr_args__': _wrapped_repr_args,
        'has_adapters': self.has_adapters,
        'prompt_template': prompt_template,
        'system_message': system_message,
      }
    ),
  )(
    runnable(backend),
    name=self.runner_name,
    embedded=False,
    models=models,
    max_batch_size=max_batch_size,
    max_latency_ms=max_latency_ms,
    scheduling_strategy=scheduling_strategy,
    runnable_init_params=dict(llm=self),
    method_configs=converter.unstructure({'generate_iterator': ModelSignature(batchable=False)}),
  )


@t.final
class LLMRunnable(bentoml.Runnable, t.Generic[M, T]):
  SUPPORTED_RESOURCES = ('amd.com/gpu', 'nvidia.com/gpu', 'cpu')
  SUPPORTS_CPU_MULTI_THREADING = True
  generate_iterator: RunnableMethod[LLMRunnable[M, T], [list[int], str, str | t.Iterable[str] | None, str | None], str]


@t.final
class LLMRunner(t.Protocol[M, T]):
  __doc__: str
  __module__: str
  llm_type: str
  llm_tag: bentoml.Tag
  identifying_params: dict[str, t.Any]
  llm: openllm.LLM[M, T]
  config: openllm.LLMConfig
  backend: LiteralBackend
  has_adapters: bool
  system_message: str | None
  prompt_template: str | None
  generate_iterator: RunnerMethod[LLMRunnable[M, T], [list[int], str, str | t.Iterable[str] | None, str | None], str]

  runner_methods: list[RunnerMethod[t.Any, t.Any, t.Any]]
  scheduling_strategy: type[Strategy]
  workers_per_resource: int | float
  runnable_init_params: dict[str, t.Any]
  _runner_handle: RunnerHandle

  def __init__(
    self,
    runnable_class: type[LLMRunnable[M, T]],
    *,
    runnable_init_params: dict[str, t.Any] | None = ...,
    name: str | None = ...,
    scheduling_strategy: type[Strategy] = ...,
    models: list[bentoml.Model] | None = ...,
    max_batch_size: int | None = ...,
    max_latency_ms: int | None = ...,
    method_configs: dict[str, dict[str, int]] | None = ...,
    embedded: bool = False,
  ) -> None: ...

  @property
  @abc.abstractmethod
  def __repr_keys__(self) -> set[str]: ...


__all__ = ['LLMRunner', 'LLMRunnable', 'LLM']
