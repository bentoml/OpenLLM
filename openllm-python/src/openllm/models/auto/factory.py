# mypy: disable-error-code="type-arg"
from __future__ import annotations
import importlib
import inspect
import logging
import typing as t
from collections import OrderedDict

import inflection

import openllm
from openllm_core.utils import ReprMixin
if t.TYPE_CHECKING:
  import types
  from collections import _odict_items
  from collections import _odict_keys
  from collections import _odict_values

  from _typeshed import SupportsIter

  from openllm_core._typing_compat import LiteralString
  from openllm_core._typing_compat import LLMRunner
  ConfigModelKeysView = _odict_keys[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]
  ConfigModelValuesView = _odict_values[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]
  ConfigModelItemsView = _odict_items[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]

logger = logging.getLogger(__name__)

class BaseAutoLLMClass:
  _model_mapping: t.ClassVar[_LazyAutoMapping]

  def __init__(self, *args: t.Any, **attrs: t.Any):
    raise EnvironmentError(f"Cannot instantiate {self.__class__.__name__} directly. Please use '{self.__class__.__name__}.Runner(model_name)' instead.")

  @classmethod
  def for_model(cls,
                model: str,
                /,
                model_id: str | None = None,
                model_version: str | None = None,
                llm_config: openllm.LLMConfig | None = None,
                ensure_available: bool = False,
                **attrs: t.Any) -> openllm.LLM[t.Any, t.Any]:
    '''The lower level API for creating a LLM instance.

    ```python
    >>> import openllm
    >>> llm = openllm.AutoLLM.for_model("flan-t5")
    ```
    '''
    llm = cls.infer_class_from_name(model).from_pretrained(model_id=model_id, model_version=model_version, llm_config=llm_config, **attrs)
    if ensure_available: llm.save_pretrained()
    return llm

  @classmethod
  def create_runner(cls, model: str, model_id: str | None = None, **attrs: t.Any) -> LLMRunner[t.Any, t.Any]:
    '''Create a LLM Runner for the given model name.

    Args:
    model: The model name to instantiate.
    model_id: The pretrained model name to instantiate.
    **attrs: Additional keyword arguments passed along to the specific configuration class.

    Returns:
    A LLM instance.
    '''
    runner_kwargs_name = set(inspect.signature(openllm.LLM[t.Any, t.Any].to_runner).parameters)
    runner_attrs = {k: v for k, v in attrs.items() if k in runner_kwargs_name}
    for k in runner_attrs:
      del attrs[k]
    return cls.for_model(model, model_id=model_id, **attrs).to_runner(**runner_attrs)

  @classmethod
  def register(cls, config_class: type[openllm.LLMConfig], llm_class: type[openllm.LLM[t.Any, t.Any]]) -> None:
    '''Register a new model for this class.

    Args:
    config_class: The configuration corresponding to the model to register.
    llm_class: The runnable to register.
    '''
    if hasattr(llm_class, 'config_class') and llm_class.config_class is not config_class:
      raise ValueError(
          f'The model class you are passing has a `config_class` attribute that is not consistent with the config class you passed (model has {llm_class.config_class} and you passed {config_class}. Fix one of those so they match!'
      )
    cls._model_mapping.register(config_class, llm_class)

  @classmethod
  def infer_class_from_name(cls, name: str) -> type[openllm.LLM[t.Any, t.Any]]:
    config_class = openllm.AutoConfig.infer_class_from_name(name)
    if config_class in cls._model_mapping: return cls._model_mapping[config_class]
    raise ValueError(
        f"Unrecognized configuration class ({config_class}) for {name}. Model name should be one of {', '.join(openllm.CONFIG_MAPPING.keys())} (Registered configuration class: {', '.join([i.__name__ for i in cls._model_mapping.keys()])})."
    )

def getattribute_from_module(module: types.ModuleType, attr: t.Any) -> t.Any:
  if attr is None: return
  if isinstance(attr, tuple): return tuple(getattribute_from_module(module, a) for a in attr)
  if hasattr(module, attr): return getattr(module, attr)
  # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the object at the top level.
  openllm_module = importlib.import_module('openllm')
  if module != openllm_module:
    try:
      return getattribute_from_module(openllm_module, attr)
    except ValueError:
      raise ValueError(f'Could not find {attr} neither in {module} nor in {openllm_module}!') from None
  raise ValueError(f'Could not find {attr} in {openllm_module}!')

class _LazyAutoMapping(OrderedDict, ReprMixin):
  """Based on transformers.models.auto.configuration_auto._LazyAutoMapping.

  This OrderedDict values() and keys() returns the list instead, so you don't
  have to do list(mapping.values()) to get the list of values.
  """
  def __init__(self, config_mapping: OrderedDict[LiteralString, LiteralString], model_mapping: OrderedDict[LiteralString, LiteralString]):
    self._config_mapping = config_mapping
    self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
    self._model_mapping = model_mapping
    self._extra_content: dict[t.Any, t.Any] = {}
    self._modules: dict[str, types.ModuleType] = {}

  def __getitem__(self, key: type[openllm.LLMConfig]) -> type[openllm.LLM[t.Any, t.Any]]:
    if key in self._extra_content: return self._extra_content[key]
    model_type = self._reverse_config_mapping[key.__name__]
    if model_type in self._model_mapping:
      return self._load_attr_from_module(model_type, self._model_mapping[model_type])
    # Maybe there was several model types associated with this config.
    model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
    for mtype in model_types:
      if mtype in self._model_mapping: return self._load_attr_from_module(mtype, self._model_mapping[mtype])
    raise KeyError(key)

  def _load_attr_from_module(self, model_type: str, attr: str) -> t.Any:
    module_name = inflection.underscore(model_type)
    if module_name not in self._modules:
      self._modules[module_name] = importlib.import_module(f'.{module_name}', 'openllm.models')
    return getattribute_from_module(self._modules[module_name], attr)

  def __len__(self) -> int:
    return len(set(self._config_mapping.keys()).intersection(self._model_mapping.keys())) + len(self._extra_content)

  @property
  def __repr_keys__(self) -> set[str]:
    return set(self._config_mapping.keys())

  def __repr__(self) -> str:
    return ReprMixin.__repr__(self)

  def __repr_args__(self) -> t.Generator[tuple[str, tuple[str, str]], t.Any, t.Any]:
    yield from ((key, (value, self._model_mapping[key])) for key, value in self._config_mapping.items() if key in self._model_mapping)

  def __bool__(self) -> bool:
    return bool(self.keys())

  def keys(self) -> ConfigModelKeysView:
    return t.cast('ConfigModelKeysView',
                  [self._load_attr_from_module(key, name) for key, name in self._config_mapping.items() if key in self._model_mapping.keys()] + list(self._extra_content.keys()))

  def values(self) -> ConfigModelValuesView:
    return t.cast('ConfigModelValuesView',
                  [self._load_attr_from_module(key, name) for key, name in self._model_mapping.items() if key in self._config_mapping.keys()] + list(self._extra_content.values()))

  def items(self) -> ConfigModelItemsView:
    return t.cast('ConfigModelItemsView',
                  [(self._load_attr_from_module(key, self._config_mapping[key]), self._load_attr_from_module(key, self._model_mapping[key]))
                   for key in self._model_mapping.keys()
                   if key in self._config_mapping.keys()] + list(self._extra_content.items()))

  def __iter__(self) -> t.Iterator[type[openllm.LLMConfig]]:
    return iter(t.cast('SupportsIter[t.Iterator[type[openllm.LLMConfig]]]', self.keys()))

  def __contains__(self, item: t.Any) -> bool:
    if item in self._extra_content: return True
    if not hasattr(item, '__name__') or item.__name__ not in self._reverse_config_mapping: return False
    return self._reverse_config_mapping[item.__name__] in self._model_mapping

  def register(self, key: t.Any, value: t.Any) -> None:
    if hasattr(key, '__name__') and key.__name__ in self._reverse_config_mapping:
      if self._reverse_config_mapping[key.__name__] in self._model_mapping.keys():
        raise ValueError(f"'{key}' is already used by a OpenLLM model.")
    self._extra_content[key] = value

__all__ = ['BaseAutoLLMClass', '_LazyAutoMapping']
