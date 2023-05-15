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
import types
import typing as t
from collections import OrderedDict

import inflection

import openllm

from .configuration_auto import AutoConfig

if t.TYPE_CHECKING:
    ConfigModelOrderedDict = OrderedDict[openllm.LLMConfig, openllm.LLM]
else:
    ConfigModelOrderedDict = OrderedDict


def _get_llm_class(config: openllm.LLMConfig, llm_mapping: _LazyAutoMapping) -> type[openllm.LLM]:
    supported_llm = llm_mapping[type(config)]
    if not isinstance(supported_llm, (list, tuple)):
        return supported_llm
    return supported_llm[0]


class _BaseAutoLLMClass:
    _model_mapping: _LazyAutoMapping

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        raise EnvironmentError(
            f"Cannot instantiate {self.__class__.__name__} directly. \
            Please use '{self.__class__.__name__}.Runner(model_name)' instead."
        )

    @t.overload
    @classmethod
    def for_model(
        cls,
        model_name: str,
        pretrained: str | None = None,
        return_runner_kwargs: t.Literal[True] = ...,
        **kwargs: t.Any,
    ) -> tuple[openllm.LLM, dict[str, t.Any]]:
        ...

    @t.overload
    @classmethod
    def for_model(
        cls,
        model_name: str,
        pretrained: str | None = None,
        return_runner_kwargs: t.Literal[False] = ...,
        **kwargs: t.Any,
    ) -> openllm.LLM:
        ...

    @classmethod
    def for_model(
        cls,
        model_name: str,
        pretrained: str | None = None,
        return_runner_kwargs: bool = True,
        **kwargs: t.Any,
    ) -> tuple[openllm.LLM, dict[str, t.Any]] | openllm.LLM:
        config = kwargs.pop("llm_config", None)
        runner_kwargs_name = [
            "name",
            "models",
            "max_batch_size",
            "max_latency_ms",
            "method_configs",
            "embedded",
            "scheduling_strategy",
        ]
        to_runner_kwargs = {k: kwargs.pop(k) for k in runner_kwargs_name if k in kwargs}
        if not isinstance(config, openllm.LLMConfig):
            # The rest of kwargs is now passed to config
            config = AutoConfig.for_model(model_name, **kwargs)
        if type(config) in cls._model_mapping.keys():
            llm = _get_llm_class(config, cls._model_mapping)(pretrained=pretrained, llm_config=config, **kwargs)
            if not return_runner_kwargs:
                return llm
            return llm, to_runner_kwargs
        raise ValueError(
            f"Unrecognized configuration class {config.__class__} for this kind of AutoRunner: {cls.__name__}.\n"
            f"Runnable type should be one of {', '.join(c.__name__ for c in cls._model_mapping.keys())}."
        )

    @classmethod
    def create_runner(cls, model_name: str, pretrained: str | None = None, **kwargs: t.Any):
        """
        Create a LLM Runner for the given model name.

        Args:
            model_name: The model name to instantiate.
            pretrained: The pretrained model name to instantiate.
            **kwargs: Additional keyword arguments passed along to the specific configuration class.

        Returns:
            A LLM instance.
        """
        llm, runner_kwargs = cls.for_model(model_name, pretrained, **kwargs)
        return llm.to_runner(**runner_kwargs)

    @classmethod
    def register(cls, config_class: type[openllm.LLMConfig], llm_class: type[openllm.LLM]):
        """
        Register a new model for this class.

        Args:
            config_class: The configuration corresponding to the model to register.
            llm_class: The runnable to register.
        """
        if hasattr(llm_class, "config_class") and llm_class.config_class != config_class:
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
    """Based on transformers.models.auto.configuration_auto._LazyAutoMapping"""

    def __init__(self, config_mapping: OrderedDict[str, str], model_mapping: OrderedDict[str, str]):
        self._config_mapping = config_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}
        self._model_mapping = model_mapping
        self._extra_content: dict[t.Any, t.Any] = {}
        self._modules: dict[str, types.ModuleType] = {}

    def __len__(self):
        common_keys = set(self._config_mapping.keys()).intersection(self._model_mapping.keys())
        return len(common_keys) + len(self._extra_content)

    def __getitem__(self, key: openllm.LLMConfig) -> openllm.LLM:
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
            self._modules[module_name] = openllm.utils.get_lazy_module(module_name)
        return getattribute_from_module(self._modules[module_name], attr)

    def keys(self):
        mapping_keys = [
            self._load_attr_from_module(key, name)
            for key, name in self._config_mapping.items()
            if key in self._model_mapping.keys()
        ]
        return mapping_keys + list(self._extra_content.keys())

    def get(self, key: openllm.LLMConfig, default: t.Any):
        try:
            return self.__getitem__(key)
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
        return mapping_values + list(self._extra_content.values())

    def items(self):
        mapping_items = [
            (
                self._load_attr_from_module(key, self._config_mapping[key]),
                self._load_attr_from_module(key, self._model_mapping[key]),
            )
            for key in self._model_mapping.keys()
            if key in self._config_mapping.keys()
        ]
        return mapping_items + list(self._extra_content.items())

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
