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
import logging
import types
import typing as t
from collections import OrderedDict

import inflection

import openllm

from .configuration_auto import AutoConfig


# NOTE: We need to do this so that overload can register
# correct overloads to typing registry
if hasattr(t, "get_overloads"):
    from typing import overload
else:
    from typing_extensions import overload


if t.TYPE_CHECKING:
    from collections import _odict_items
    from collections import _odict_keys
    from collections import _odict_values

    from ..._llm import LLMRunner

    ConfigModelOrderedDict = OrderedDict[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]
    ConfigModelKeysView = _odict_keys[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]
    ConfigModelValuesView = _odict_values[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]
    ConfigModelItemsView = _odict_items[type[openllm.LLMConfig], type[openllm.LLM[t.Any, t.Any]]]
else:
    ConfigModelKeysView = ConfigModelValuesView = ConfigModelItemsView = t.Any
    ConfigModelOrderedDict = OrderedDict

logger = logging.getLogger(__name__)


class _BaseAutoLLMClass:
    _model_mapping: _LazyAutoMapping

    def __init__(self, *args: t.Any, **attrs: t.Any):
        raise EnvironmentError(
            f"Cannot instantiate {self.__class__.__name__} directly. "
            "Please use '{self.__class__.__name__}.Runner(model_name)' instead."
        )

    @overload
    @classmethod
    def for_model(
        cls,
        model_name: str,
        model_id: str | None = None,
        return_runner_kwargs: t.Literal[False] = ...,
        llm_config: openllm.LLMConfig | None = ...,
        ensure_available: t.Literal[False, True] = ...,
        **attrs: t.Any,
    ) -> openllm.LLM[t.Any, t.Any]:
        ...

    @overload
    @classmethod
    def for_model(
        cls,
        model_name: str,
        model_id: str | None = None,
        return_runner_kwargs: t.Literal[True] = ...,
        llm_config: openllm.LLMConfig | None = ...,
        ensure_available: t.Literal[False, True] = ...,
        **attrs: t.Any,
    ) -> tuple[openllm.LLM[t.Any, t.Any], dict[str, t.Any]]:
        ...

    @classmethod
    def for_model(
        cls,
        model_name: str,
        model_id: str | None = None,
        return_runner_kwargs: bool = False,
        llm_config: openllm.LLMConfig | None = None,
        ensure_available: bool = False,
        **attrs: t.Any,
    ) -> openllm.LLM[t.Any, t.Any] | tuple[openllm.LLM[t.Any, t.Any], dict[str, t.Any]]:
        """The lower level API for creating a LLM instance.

        ```python
        >>> import openllm
        >>> llm = openllm.AutoLLM.for_model("flan-t5")
        ```
        """
        # order matters here
        runner_kwargs_name = {
            "models",
            "max_batch_size",
            "max_latency_ms",
            "method_configs",
            "scheduling_strategy",
        }
        to_runner_attrs = {k: v for k, v in attrs.items() if k in runner_kwargs_name}
        attrs = {k: v for k, v in attrs.items() if k not in to_runner_attrs}
        if cls._model_mapping.get(inflection.underscore(model_name), None, mapping_type="name2model"):
            if not isinstance(llm_config, openllm.LLMConfig):
                # The rest of kwargs is now passed to config
                llm_config = AutoConfig.for_model(model_name, **attrs)
                attrs = llm_config.__openllm_extras__
            # the rest of attrs will be saved to __openllm_extras__
            llm = cls._model_mapping[type(llm_config)].from_pretrained(
                model_id,
                llm_config=llm_config,
                **attrs,
            )
            if ensure_available:
                logger.debug(
                    "'ensure_available=True', Downloading '%s' with 'model_id=%s' to local model store.",
                    model_name,
                    llm.model_id,
                )
                llm.ensure_model_id_exists()
            if not return_runner_kwargs:
                return llm
            return llm, to_runner_attrs
        raise ValueError(
            f"Unrecognized configuration class {llm_config.__class__} for this kind of AutoLLM: {cls.__name__}.\n"
            f"LLM type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def create_runner(cls, model_name: str, model_id: str | None = None, **attrs: t.Any) -> LLMRunner:
        """
        Create a LLM Runner for the given model name.

        Args:
            model_name: The model name to instantiate.
            model_id: The pretrained model name to instantiate.
            **attrs: Additional keyword arguments passed along to the specific configuration class.

        Returns:
            A LLM instance.
        """
        llm, runner_attrs = cls.for_model(model_name, model_id, return_runner_kwargs=True, **attrs)
        return llm.to_runner(**runner_attrs)

    @classmethod
    def register(cls, config_class: type[openllm.LLMConfig], llm_class: type[openllm.LLM[t.Any, t.Any]]):
        """
        Register a new model for this class.

        Args:
            config_class: The configuration corresponding to the model to register.
            llm_class: The runnable to register.
        """
        if hasattr(llm_class, "config_class") and llm_class.config_class is not config_class:
            raise ValueError(
                "The model class you are passing has a `config_class` attribute that is not consistent with the "
                f"config class you passed (model has {llm_class.config_class} and you passed {config_class}. Fix "
                "one of those so they match!"
            )
        cls._model_mapping.register(config_class, llm_class)


def getattribute_from_module(module: types.ModuleType, attr: t.Any) -> t.Any:
    if attr is None:
        return None
    if isinstance(attr, tuple):
        return tuple(getattribute_from_module(module, a) for a in attr)
    if hasattr(module, attr):
        return getattr(module, attr)
    # Some of the mappings have entries model_type -> object of another model type. In that case we try to grab the
    # object at the top level.
    openllm_module = importlib.import_module("openllm")

    if module != openllm_module:
        try:
            return getattribute_from_module(openllm_module, attr)
        except ValueError:
            raise ValueError(f"Could not find {attr} neither in {module} nor in {openllm_module}!")
    else:
        raise ValueError(f"Could not find {attr} in {openllm_module}!")


class _LazyAutoMapping(ConfigModelOrderedDict):
    """Based on transformers.models.auto.configuration_auto._LazyAutoMapping
    This OrderedDict values() and keys() returns the list instead, so you don't
    have to do list(mapping.values()) to get the list of values.
    """

    def __init__(self, config_mapping: OrderedDict[str, str], model_mapping: OrderedDict[str, str]):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._extra_content: dict[t.Any, t.Any] = {}
        self._modules: dict[str, types.ModuleType] = {}

    def __len__(self):
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key: type[openllm.LLMConfig]) -> type[openllm.LLM[t.Any, t.Any]]:
        if key in self._extra_content:
            return self._extra_content[key]
        model_type = self._reverse_config_mapping[key.__name__]
        if model_type in self._model_mapping:
            model_name = self._model_mapping[model_type]
            return self._load_attr_from_module(model_type, model_name)

        # Maybe there was several model types associated with this config.
        model_types = [k for k, v in self._config_mapping.items() if v == key.__name__]
        for mtype in model_types:
            if mtype in self._model_mapping:
                model_name = self._model_mapping[mtype]
                return self._load_attr_from_module(mtype, model_name)
        raise KeyError(key)

    def _load_attr_from_module(self, model_type: str, attr: str) -> t.Any:
        module_name = inflection.underscore(model_type)
        if module_name not in self._modules:
            self._modules[module_name] = importlib.import_module(f".{module_name}", "openllm.models")
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return t.cast(ConfigModelKeysView, mapping_keys + list(self._extra_content.keys()))

    @overload
    def get(
        self, key: type[openllm.LLMConfig], default: t.Any, mapping_type: t.Literal["default"] = "default"
    ) -> type[openllm.LLM[t.Any, t.Any]]:
        ...

    @overload
    def get(self, key: str, default: t.Any, mapping_type: t.Literal["name2model", "name2config"] = ...) -> str:
        ...

    def get(
        self,
        key: str | type[openllm.LLMConfig],
        default: t.Any,
        mapping_type: t.Literal["default", "name2config", "name2model"] = "default",
    ) -> str | type[openllm.LLM[t.Any, t.Any]]:
        _supported = {"default", "name2model", "name2config"}
        if mapping_type not in _supported:
            raise RuntimeError(f"Unknown mapping type {mapping_type} (supported: {_supported})")

        if mapping_type == "default":
            if t.TYPE_CHECKING:
                # we check for lenient_issubclass below, but pyright is too dumb to understand
                assert not isinstance(key, str)
            else:
                if not openllm.utils.lenient_issubclass(key, openllm.LLMConfig):
                    raise KeyError(f"Key must be a type of 'openllm.LLMConfig', got {key} instead.")
            try:
                return self.__getitem__(key)
            except KeyError:
                return default
        else:
            mapping = self._model_mapping if mapping_type == "name2model" else self._config_mapping
            assert isinstance(key, str), f"Key must be a string type if mapping_type={mapping_type}"
            try:
                return mapping.__getitem__(key)
            except KeyError:
                return default

    def __bool__(self):
        return bool(self.keys())

    def values(self):
        mapping_values = [
            self._load_attr_from_module(key, name)
            for key, name in self._model_mapping.items()
            if key in self._config_mapping.keys()
        ]
        return t.cast(ConfigModelValuesView, mapping_values + list(self._extra_content.values()))

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return t.cast(ConfigModelItemsView, mapping_items + list(self._extra_content.items()))

    def __iter__(self):
        return iter(self.keys())

    def __contains__(self, item: t.Any):
        if item in self._extra_content:
            return True
        if not hasattr(item, "__name__") or item.__name__ not in self._reverse_config_mapping:
            return False
        model_type = self._reverse_config_mapping[item.__name__]
        return model_type in self._model_mapping

    def register(self, key: t.Any, value: t.Any):
        """
        Register a new model in this mapping.
        """
        if hasattr(key, "__name__") and key.__name__ in self._reverse_config_mapping:
            model_type = self._reverse_config_mapping[key.__name__]
            if model_type in self._model_mapping.keys():
                raise ValueError(f"'{key}' is already used by a OpenLLM model.")

        self._extra_content[key] = value


__all__ = ["_BaseAutoLLMClass", "_LazyAutoMapping"]
