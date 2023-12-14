from __future__ import annotations
import importlib, os, typing as t
from collections import OrderedDict
import inflection, orjson
from openllm_core.exceptions import MissingDependencyError, OpenLLMException
from openllm_core.utils import ReprMixin, is_bentoml_available
from openllm_core.utils.import_utils import is_transformers_available

if t.TYPE_CHECKING:
  import types
  from collections import _odict_items, _odict_keys, _odict_values

  import openllm, openllm_core
  from openllm_core._typing_compat import LiteralString, M, T

  ConfigKeysView = _odict_keys[str, type[openllm_core.LLMConfig]]
  ConfigValuesView = _odict_values[str, type[openllm_core.LLMConfig]]
  ConfigItemsView = _odict_items[str, type[openllm_core.LLMConfig]]
  OrderedDictType = OrderedDict[LiteralString, type[openllm_core.LLMConfig]]
else:
  OrderedDictType = OrderedDict

# NOTE: This is the entrypoint when adding new model config
CONFIG_MAPPING_NAMES = OrderedDict(
  sorted([
    ('flan_t5', 'FlanT5Config'),
    ('baichuan', 'BaichuanConfig'),
    ('chatglm', 'ChatGLMConfig'),
    ('falcon', 'FalconConfig'),
    ('gpt_neox', 'GPTNeoXConfig'),
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
    if key in self._mapping.keys():
      raise ValueError(f"'{key}' is already used by a OpenLLM config, pick another name.")
    self._extra_content[key] = value


CONFIG_MAPPING: dict[LiteralString, type[openllm_core.LLMConfig]] = _LazyConfigMapping(CONFIG_MAPPING_NAMES)
# The below handle special alias when we call underscore to the name directly without processing camelcase first.
CONFIG_NAME_ALIASES: dict[str, str] = {'chat_glm': 'chatglm', 'stable_lm': 'stablelm', 'star_coder': 'starcoder', 'gpt_neo_x': 'gpt_neox'}
CONFIG_FILE_NAME = 'config.json'


class AutoConfig:
  def __init__(self, *_: t.Any, **__: t.Any):
    raise EnvironmentError('Cannot instantiate AutoConfig directly. Use `.for_model(model_name)`')

  # update-config-stubs.py: auto stubs start
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['baichuan'], **attrs: t.Any) -> openllm_core.config.BaichuanConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['chatglm'], **attrs: t.Any) -> openllm_core.config.ChatGLMConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['dolly_v2'], **attrs: t.Any) -> openllm_core.config.DollyV2Config: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['falcon'], **attrs: t.Any) -> openllm_core.config.FalconConfig: ...
  @t.overload
  @classmethod
  def for_model(cls, model_name: t.Literal['flan_t5'], **attrs: t.Any) -> openllm_core.config.FlanT5Config: ...
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
  # update-config-stubs.py: auto stubs stop

  @classmethod
  def for_model(cls, model_name: str, **attrs: t.Any) -> openllm_core.LLMConfig:
    model_name = inflection.underscore(model_name)
    if model_name in CONFIG_MAPPING:
      return CONFIG_MAPPING[model_name].model_construct_env(**attrs)
    raise ValueError(f"Unrecognized configuration class for {model_name}. Model name should be one of {', '.join(CONFIG_MAPPING.keys())}.")

  @classmethod
  def infer_class_from_name(cls, name: str) -> type[openllm_core.LLMConfig]:
    model_name = inflection.underscore(name)
    if model_name in CONFIG_NAME_ALIASES:
      model_name = CONFIG_NAME_ALIASES[model_name]
    if model_name in CONFIG_MAPPING:
      return CONFIG_MAPPING[model_name]
    raise ValueError(f"Unrecognized configuration class for {model_name}. Model name should be one of {', '.join(CONFIG_MAPPING.keys())}.")

  _cached_mapping = None

  @classmethod
  def _CONFIG_MAPPING_NAMES_TO_ARCHITECTURE(cls) -> dict[str, str]:
    if cls._cached_mapping is None:
      AutoConfig._cached_mapping = {v.__config__['architecture']: k for k, v in CONFIG_MAPPING.items()}
    return AutoConfig._cached_mapping

  @classmethod
  def infer_class_from_llm(cls, llm: openllm.LLM[M, T]) -> type[openllm_core.LLMConfig]:
    if not is_bentoml_available():
      raise MissingDependencyError("Requires 'bentoml' to be available. Do 'pip install bentoml'")
    if llm._local:
      config_file = os.path.join(llm.model_id, CONFIG_FILE_NAME)
    else:
      try:
        config_file = llm.bentomodel.path_of(CONFIG_FILE_NAME)
      except OpenLLMException as err:
        if not is_transformers_available():
          raise MissingDependencyError("Requires 'transformers' to be available. Do 'pip install transformers'") from err
        from transformers.utils import cached_file

        try:
          config_file = cached_file(llm.model_id, CONFIG_FILE_NAME)
        except Exception as err:
          raise ValueError(
            "Failed to determine architecture from 'config.json'. If this is a gated model, make sure to pass in HUGGING_FACE_HUB_TOKEN"
          ) from err
    if not os.path.exists(config_file):
      raise ValueError(f"Failed to find 'config.json' (config_json_path={config_file})")
    with open(config_file, 'r', encoding='utf-8') as f:
      loaded_config = orjson.loads(f.read())
    if 'architectures' in loaded_config:
      for architecture in loaded_config['architectures']:
        if architecture in cls._CONFIG_MAPPING_NAMES_TO_ARCHITECTURE():
          return cls.infer_class_from_name(cls._CONFIG_MAPPING_NAMES_TO_ARCHITECTURE()[architecture])
    raise ValueError(f"Failed to determine config class for '{llm.model_id}'. Make sure {llm.model_id} is saved with openllm.")
