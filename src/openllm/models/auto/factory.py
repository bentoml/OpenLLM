# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations
import importlib
import inspect
import logging
import sys
import typing as t
from collections import OrderedDict
import inflection
import openllm
from .configuration_auto import AutoConfig
from ...utils import ReprMixin
# NOTE: We need to do this so that overload can register
# correct overloads to typing registry
if sys.version_info[:2] >= (3, 11): from typing import overload
else: from typing_extensions import overload
if t.TYPE_CHECKING:
    import types
    from ..._llm import LLMRunner
    from collections import _odict_items, _odict_keys, _odict_values
    ConfigModelOrderedDict = OrderedDict[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]
    ConfigModelKeysView = _odict_keys[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]
    ConfigModelValuesView = _odict_values[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]
    ConfigModelItemsView = _odict_items[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]
else:
    ConfigModelKeysView = ConfigModelValuesView = ConfigModelItemsView = t.Any
    ConfigModelOrderedDict = OrderedDict
logger = logging.getLogger(__name__)
class BaseAutoLLMClass:
    _model_mapping: _LazyAutoMapping
    def __init__(self, *args: t.Any, **attrs: t.Any): raise EnvironmentError(f"Cannot instantiate {self.__class__.__name__} directly. Please use '{self.__class__.__name__}.Runner(model_name)' instead.")
    @overload
    @classmethod
    def for_model(cls, model: str, model_id: str | None = None, model_version: str | None = None, return_runner_kwargs: t.Literal[False] = False, llm_config: openllm.LLMConfig | None = ..., ensure_available: t.Literal[False, True] = ..., **attrs: t.Any) -> openllm.LLM[t.Any, t.Any]: ...
    @overload
    @classmethod
    def for_model(cls, model: str, model_id: str | None = None, model_version: str | None = None, return_runner_kwargs: t.Literal[True] = ..., llm_config: openllm.LLMConfig | None = ..., ensure_available: t.Literal[False, True] = ..., **attrs: t.Any) -> tuple[openllm.LLM[t.Any, t.Any], dict[str, t.Any]]: ...
    @classmethod
    def for_model(cls, model: str, model_id: str | None = None, model_version: str | None = None, return_runner_kwargs: bool = False, llm_config: openllm.LLMConfig | None = None, ensure_available: bool = False, **attrs: t.Any) -> openllm.LLM[t.Any, t.Any] | tuple[openllm.LLM[t.Any, t.Any], dict[str, t.Any]]:
        """The lower level API for creating a LLM instance.

        ```python
        >>> import openllm
        >>> llm = openllm.AutoLLM.for_model("flan-t5")
        ```

        To return the runner kwargs instead of the LLM instance, set `return_runner_kwargs=True`:
        ```python
        >>> import openllm_module
        >>> llm, runner_kwargs = openllm.AutoLLM.for_model("flan-t5", return_runner_kwargs=True)
        >>> runner = llm.to_runner(**runner_kwargs)
        ```
        """
        runner_kwargs_name = set(inspect.signature(openllm.LLM[t.Any, t.Any].to_runner).parameters)
        to_runner_attrs = {k: v for k, v in attrs.items() if k in runner_kwargs_name}
        attrs = {k: v for k, v in attrs.items() if k not in to_runner_attrs}
        if not isinstance(llm_config, openllm.LLMConfig):
            # The rest of kwargs is now passed to config
            llm_config = AutoConfig.for_model(model, **attrs)
            attrs = llm_config.__openllm_extras__
        # the rest of attrs will be saved to __openllm_extras__
        if type(llm_config) in cls._model_mapping.keys():
            model_class = cls._model_mapping[type(llm_config)]
            llm = model_class.from_pretrained(model_id, model_version=model_version, llm_config=llm_config, **attrs)
            if ensure_available: llm.ensure_model_id_exists()
            if not return_runner_kwargs: return llm
            return llm, to_runner_attrs
        raise ValueError(f"Unrecognized configuration class {llm_config.__class__} for this kind of AutoLLM: {cls.__name__}.\nLLM type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}.")
    @classmethod
    def create_runner(cls, model: str, model_id: str | None = None, **attrs: t.Any) -> LLMRunner[t.Any, t.Any]:
        """Create a LLM Runner for the given model name.

        Args:
            model: The model name to instantiate.
            model_id: The pretrained model name to instantiate.
            **attrs: Additional keyword arguments passed along to the specific configuration class.

        Returns:
            A LLM instance.
        """
        llm, runner_attrs = cls.for_model(model, model_id, return_runner_kwargs=True, **attrs)
        return llm.to_runner(**runner_attrs)
    @classmethod
    def register(cls, config_class: type[openllm.LLMConfig], llm_class: type[openllm.LLM[t.Any, t.Any]]):
        """Register a new model for this class.

        Args:
            config_class: The configuration corresponding to the model to register.
            llm_class: The runnable to register.
        """
        if hasattr(llm_class, "config_class") and llm_class.config_class is not config_class: raise ValueError("The model class you are passing has a `config_class` attribute that is not consistent with the config class you passed (model has {llm_class.config_class} and you passed {config_class}. Fix one of those so they match!")
        cls._model_mapping.register(config_class, llm_class)
def getattribute_from_module(module: types.ModuleType, attr: t.Any) -> t.Any:
    if attr is None: return
    if isinstance(attr, tuple): return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr): return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    openllm_module = importlib.import_module("openllm")
    if module != openllm_module:
        try: return getattribute_from_module(openllm_module, attr)
        except ValueError: raise ValueError(f"Could not find {attr} neither in {module} nor in {openllm_module}!") from None
    else: raise ValueError(f"Could not find {attr} in {openllm_module}!")
class _LazyAutoMapping(ConfigModelOrderedDict, ReprMixin):
    """Based on transformers.models.auto.configuration_auto._LazyAutoMapping.

    This OrderedDict values() and keys() returns the list instead, so you don't
    have to do list(mapping.values()) to get the list of values.
    """
    def __init__(self, config_mapping: OrderedDict[t.LiteralString, t.LiteralString], model_mapping: OrderedDict[t.LiteralString, t.LiteralString]):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._extra_content: dict[t.Any, t.Any] = {}
        self._modules: dict[str, types.ModuleType] = {}
    def __len__(self): return len(set(self._config_mapping.keys()).intersection(self._model_mapping.keys())) + len(self._extra_content)
    def __getitem__(self, key: type[openllm.LLMConfig]) -> type[openllm.LLM[t.Any, t.Any]]:
        if key in self._extra_content: return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping: return self._load_attr_from_module(model_type, self._model_mapping[model_type])
        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping: return self._load_attr_from_module(mtype, self._model_mapping[mtype])
        raise KeyError(key)
    def _load_attr_from_module(self, model_type: str, attr: str) -> t.Any:
        module_name = inflection.underscore(model_type)
        if module_name not in self._modules: self._modules[module_name] = importlib.import_module(f".{module_name}", "openllm.models")
        return getattribute_from_module(self._modules[module_name], attr)
    def keys(self): return t.cast(ConfigModelKeysView, [self._load_attr_from_module(key, name) for key, name in self._config_mapping.items() if key in self._model_mapping.keys()] + list(self._extra_content.keys()))
    @property
    def __repr_keys__(self) -> set[str]: return set(self._config_mapping.keys())
    def __repr__(self) -> str: return ReprMixin.__repr__(self)
    def __repr_args__(self) -> t.Generator[tuple[str, tuple[str, str]], t.Any, t.Any]: yield from ((key, (value, self._model_mapping[key])) for key, value in self._config_mapping.items() if key in self._model_mapping)
    def __bool__(self): return bool(self.keys())
    def values(self): return t.cast(ConfigModelValuesView, [self._load_attr_from_module(key, name) for key, name in self._model_mapping.items() if key in self._config_mapping.keys()] + list(self._extra_content.values()))
    def items(self): return t.cast(ConfigModelItemsView, [(self._load_attr_from_module(key, self._config_mapping[key]), self._load_attr_from_module(key, self._model_mapping[key])) for key in self._model_mapping.keys() if key in self._config_mapping.keys()] + list(self._extra_content.items()))
    def __iter__(self): return iter(self.keys())
    def __contains__(self, item: t.Any):
        if item in self._extra_content: return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping: return False
        return self._reverse_config_mapping[item.__name__] in self._model_mapping
    def register(self, key: t.Any, value: t.Any):
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            if self._reverse_config_mapping[key.__name__] in self._model_mapping.keys(): raise ValueError(f"'{key}' is already used by a OpenLLM model.")
        self._extra_content[key] = value
__all__ = ["BaseAutoLLMClass", "_LazyAutoMapping"]
