# mypy: disable-error-code="type-arg"
from __future__ import annotations
import importlib
import typing as t

from collections import OrderedDict

import inflection

import openllm_core

from openllm_core.utils import ReprMixin

if t.TYPE_CHECKING:
  import types

  from collections import _odict_items
  from collections import _odict_keys
  from collections import _odict_values

  from openllm_core._typing_compat import LiteralString
  ConfigKeysView = _odict_keys[str, type[openllm_core.LLMConfig]]
  ConfigValuesView = _odict_values[str, type[openllm_core.LLMConfig]]
  ConfigItemsView = _odict_items[str, type[openllm_core.LLMConfig]]

# NOTE: This is the entrypoint when adding new model config
CONFIG_MAPPING_NAMES = OrderedDict([('chatglm', 'ChatGLMConfig'), ('dolly_v2', 'DollyV2Config'), ('falcon', 'FalconConfig'), ('flan_t5', 'FlanT5Config'), ('gpt_neox', 'GPTNeoXConfig'),
                                    ('llama', 'LlamaConfig'), ('mpt', 'MPTConfig'), ('opt', 'OPTConfig'), ('stablelm', 'StableLMConfig'), ('starcoder', 'StarCoderConfig'),
                                    ('baichuan', 'BaichuanConfig')])

class _LazyConfigMapping(OrderedDict, ReprMixin):
  def __init__(self, mapping: OrderedDict[LiteralString, LiteralString]):
    self._mapping = mapping
    self._extra_content: dict[str, t.Any] = {}
    self._modules: dict[str, types.ModuleType] = {}

  def __getitem__(self, key: str) -> t.Any:
    if key in self._extra_content: return self._extra_content[key]
    if key not in self._mapping:
      if inflection.underscore(key) in self._mapping: return self.__getitem__(inflection.underscore(key))
      raise KeyError(key)
    value, module_name = self._mapping[key], inflection.underscore(key)
    if module_name not in self._modules: self._modules[module_name] = openllm_core.utils.EnvVarMixin(module_name).module
    if hasattr(self._modules[module_name], value): return getattr(self._modules[module_name], value)
    # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the object at the top level.
    return getattr(importlib.import_module('openllm'), value)

  @property
  def __repr_keys__(self) -> set[str]:
    return set(self._mapping.keys())

  def __repr__(self) -> str:
    return ReprMixin.__repr__(self)

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
    if key in self._mapping.keys(): raise ValueError(f"'{key}' is already used by a OpenLLM config, pick another name.")
    self._extra_content[key] = value

CONFIG_MAPPING: dict[str, type[openllm_core.LLMConfig]] = _LazyConfigMapping(CONFIG_MAPPING_NAMES)
# The below handle special alias when we call underscore to the name directly without processing camelcase first.
CONFIG_NAME_ALIASES: dict[str, str] = {'chat_glm': 'chatglm', 'stable_lm': 'stablelm', 'star_coder': 'starcoder', 'gpt_neo_x': 'gpt_neox'}

class AutoConfig:
  def __init__(self, *_: t.Any, **__: t.Any):
    raise EnvironmentError('Cannot instantiate AutoConfig directly. Please use `AutoConfig.for_model(model_name)` instead.')

  @classmethod
  def for_model(cls, model_name: str, **attrs: t.Any) -> openllm_core.LLMConfig:
    model_name = inflection.underscore(model_name)
    if model_name in CONFIG_MAPPING: return CONFIG_MAPPING[model_name].model_construct_env(**attrs)
    raise ValueError(f"Unrecognized configuration class for {model_name}. Model name should be one of {', '.join(CONFIG_MAPPING.keys())}.")

  @classmethod
  def infer_class_from_name(cls, name: str) -> type[openllm_core.LLMConfig]:
    model_name = inflection.underscore(name)
    if model_name in CONFIG_NAME_ALIASES: model_name = CONFIG_NAME_ALIASES[model_name]
    if model_name in CONFIG_MAPPING: return CONFIG_MAPPING[model_name]
    raise ValueError(f"Unrecognized configuration class for {model_name}. Model name should be one of {', '.join(CONFIG_MAPPING.keys())}.")
