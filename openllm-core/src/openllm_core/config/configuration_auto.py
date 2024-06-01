from __future__ import annotations

import importlib, typing as t, inflection
from collections import OrderedDict
from ..utils import ReprMixin

if t.TYPE_CHECKING:
  import types
  from collections import _odict_items, _odict_keys, _odict_values

  import openllm, openllm_core
  from openllm_core._typing_compat import LiteralString

  ConfigKeysView = _odict_keys[str, type[openllm_core.LLMConfig]]
  ConfigValuesView = _odict_values[str, type[openllm_core.LLMConfig]]
  ConfigItemsView = _odict_items[str, type[openllm_core.LLMConfig]]
  OrderedDictType = OrderedDict[LiteralString, type[openllm_core.LLMConfig]]
else:
  OrderedDictType = OrderedDict

ModelType: t.TypeAlias = t.Literal[
  'baichuan',
  'chatglm',
  'commandr',
  'falcon',
  'gemma',
  'gpt_neox',
  'dolly_v2',
  'dbrx',
  'stablelm',
  'llama',
  'mpt',
  'opt',
  'phi',
  'qwen',
  'starcoder',
  'mistral',
  'mixtral',
  'yi',
]

# NOTE: This is the entrypoint when adding new model config
CONFIG_MAPPING_NAMES: OrderedDict[ModelType, str] = OrderedDict(
  sorted([
    ('baichuan', 'BaichuanConfig'),
    ('chatglm', 'ChatGLMConfig'),
    ('commandr', 'CohereConfig'),
    ('falcon', 'FalconConfig'),
    ('gpt_neox', 'GPTNeoXConfig'),
    ('gemma', 'GemmaConfig'),
    ('dbrx', 'DbrxConfig'),
    ('dolly_v2', 'DollyV2Config'),
    ('stablelm', 'StableLMConfig'),
    ('llama', 'LlamaConfig'),
    ('mpt', 'MPTConfig'),
    ('opt', 'OPTConfig'),
    ('phi', 'PhiConfig'),
    ('qwen', 'QwenConfig'),
    ('starcoder', 'StarCoderConfig'),
    ('mistral', 'MistralConfig'),
    ('mixtral', 'MixtralConfig'),
    ('yi', 'YiConfig'),
  ])
)
CONFIG_TO_ALIAS_NAMES: OrderedDict[str, ModelType] = OrderedDict({v: k for k, v in CONFIG_MAPPING_NAMES.items()})
M = t.TypeVar('M')
T = t.TypeVar('T')


class _LazyConfigMapping(OrderedDictType, ReprMixin):
  def __init__(self, mapping: OrderedDict[LiteralString, LiteralString]):
    self._mapping = mapping
    self._extra_content: dict[str, t.Any] = {}
    self._modules: dict[str, types.ModuleType] = {}

  def __getitem__(self, key: str) -> t.Any:
    if key in self._extra_content:
      return self._extra_content[key]
    if key not in self._mapping:
      if inflection.underscore(key) in self._mapping:
        return self.__getitem__(inflection.underscore(key))
      raise KeyError(key)
    value, module_name = self._mapping[key], inflection.underscore(key)
    if module_name not in self._modules:
      self._modules[module_name] = importlib.import_module(f'.configuration_{module_name}', 'openllm_core.config')
    if hasattr(self._modules[module_name], value):
      return getattr(self._modules[module_name], value)
    # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the object at the top level.
    return getattr(importlib.import_module('openllm'), value)

  @property
  def __repr_keys__(self) -> set[str]:
    return set(self._mapping.keys())

  def __repr_args__(self) -> t.Generator[tuple[str, t.Any], t.Any, t.Any]:
    yield from self._mapping.items()

  def keys(self) -> ConfigKeysView:
    return t.cast('ConfigKeysView', list(self._mapping.keys()) + list(self._extra_content.keys()))

  def values(self) -> ConfigValuesView:
    return t.cast('ConfigValuesView', [self[k] for k in self._mapping.keys()] + list(self._extra_content.values()))

  def items(self) -> ConfigItemsView:
    return t.cast('ConfigItemsView', [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items()))

  def __iter__(self) -> t.Iterator[str]:
    return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

  def __contains__(self, item: t.Any) -> bool:
    return item in self._mapping or item in self._extra_content

  def register(self, key: str, value: t.Any) -> None:
    if key in self._mapping.keys():
      raise ValueError(f"'{key}' is already used by a OpenLLM config, pick another name.")
    self._extra_content[key] = value


CONFIG_MAPPING: dict[ModelType, type[openllm_core.LLMConfig]] = _LazyConfigMapping(CONFIG_MAPPING_NAMES)
CONFIG_FILE_NAME = 'config.json'


class AutoConfig:
  def __init__(self, *_: t.Any, **__: t.Any):
    raise EnvironmentError('Cannot instantiate AutoConfig directly. Use `.for_model(model_name)`')

  # fmt: off
  # update-config-stubs.py: auto stubs start
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['baichuan'], **attrs: t.Any) -> openllm_core.config.BaichuanConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['chatglm'], **attrs: t.Any) -> openllm_core.config.ChatGLMConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['commandr'], **attrs: t.Any) -> openllm_core.config.CohereConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['dbrx'], **attrs: t.Any) -> openllm_core.config.DbrxConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['dolly_v2'], **attrs: t.Any) -> openllm_core.config.DollyV2Config: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['falcon'], **attrs: t.Any) -> openllm_core.config.FalconConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['gemma'], **attrs: t.Any) -> openllm_core.config.GemmaConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['gpt_neox'], **attrs: t.Any) -> openllm_core.config.GPTNeoXConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['llama'], **attrs: t.Any) -> openllm_core.config.LlamaConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['mistral'], **attrs: t.Any) -> openllm_core.config.MistralConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['mixtral'], **attrs: t.Any) -> openllm_core.config.MixtralConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['mpt'], **attrs: t.Any) -> openllm_core.config.MPTConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['opt'], **attrs: t.Any) -> openllm_core.config.OPTConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['phi'], **attrs: t.Any) -> openllm_core.config.PhiConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['qwen'], **attrs: t.Any) -> openllm_core.config.QwenConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['stablelm'], **attrs: t.Any) -> openllm_core.config.StableLMConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['starcoder'], **attrs: t.Any) -> openllm_core.config.StarCoderConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['yi'], **attrs: t.Any) -> openllm_core.config.YiConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: LiteralString, **attrs: t.Any) -> openllm_core.LLMConfig: ...
  # update-config-stubs.py: auto stubs stop
  # fmt: on
  @classmethod
  def for_model(cls, model_name: str, **attrs: t.Any) -> openllm_core.LLMConfig:
    model_name = inflection.underscore(model_name)
    if model_name in CONFIG_MAPPING:
      return CONFIG_MAPPING[model_name].model_construct_env(**attrs)
    raise ValueError(
      f"Unrecognized configuration class for {model_name}. Model name should be one of {', '.join(CONFIG_MAPPING.keys())}."
    )

  _architecture_mappings = {it().metadata_config['architecture']: k for k, it in CONFIG_MAPPING.items()}

  @classmethod
  def from_llm(cls, llm: openllm.LLM, **attrs: t.Any) -> openllm_core.LLMConfig:
    config_cls = llm.config.__class__.__name__
    if config_cls in CONFIG_TO_ALIAS_NAMES:
      return cls.from_id(llm.model_id, trust_remote_code=llm.trust_remote_code, **attrs)
    raise ValueError(
      f"Failed to determine config class for '{llm.model_id}'. Make sure {llm.model_id} is saved with openllm."
    )

  @classmethod
  def from_id(cls, model_id: str, *, trust_remote_code: bool = False, **attrs: t.Any) -> openllm_core.LLMConfig:
    import transformers

    config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    for arch in config.architectures:
      if arch in cls._architecture_mappings:
        return cls.for_model(cls._architecture_mappings[arch]).model_construct_env(**attrs)
    else:
      raise RuntimeError(
        f'Failed to determine config class for {model_id}. Got {config.architectures}, which is not yet supported (Supported: {list(cls._architecture_mappings.keys())})'
      )
