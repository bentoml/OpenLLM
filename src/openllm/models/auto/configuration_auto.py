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
import typing as t
from collections import OrderedDict

import inflection

import openllm

from ...utils import ReprMixin


if t.TYPE_CHECKING:
    import types
    from collections import _odict_items
    from collections import _odict_keys
    from collections import _odict_values

    ConfigOrderedDict = OrderedDict[str, type[openllm.LLMConfig]]

    ConfigKeysView = _odict_keys[str, type[openllm.LLMConfig]]
    ConfigValuesView = _odict_values[str, type[openllm.LLMConfig]]
    ConfigItemsView = _odict_items[str, type[openllm.LLMConfig]]
else:
    ConfigKeysView = ConfigValuesView = ConfigItemsView = t.Any
    ConfigOrderedDict = OrderedDict

# NOTE: This is the entrypoint when adding new model config
CONFIG_MAPPING_NAMES = OrderedDict(
    [
        ("chatglm", "ChatGLMConfig"),
        ("dolly_v2", "DollyV2Config"),
        ("falcon", "FalconConfig"),
        ("flan_t5", "FlanT5Config"),
        ("gpt_neox", "GPTNeoXConfig"),
        ("llama", "LlaMAConfig"),
        ("mpt", "MPTConfig"),
        ("opt", "OPTConfig"),
        ("stablelm", "StableLMConfig"),
        ("starcoder", "StarCoderConfig"),
        ("baichuan", "BaichuanConfig"),
    ]
)


class _LazyConfigMapping(ConfigOrderedDict, ReprMixin):
    def __init__(self, mapping: OrderedDict[t.LiteralString, t.LiteralString]):
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
        value = self._mapping[key]
        module_name = inflection.underscore(key)
        if module_name not in self._modules:
            self._modules[module_name] = openllm.utils.EnvVarMixin(module_name).module
        if hasattr(self._modules[module_name], value):
            return getattr(self._modules[module_name], value)

        # Some of the mappings have entries model_type -> config of another model type. In that case we try to grab the
        # object at the top level.
        return getattr(openllm, value)

    @property
    def __repr_keys__(self) -> set[str]:
        return set(self._mapping.keys())

    def __repr__(self) -> str:
        return ReprMixin.__repr__(self)

    def __repr_args__(self) -> t.Generator[tuple[str, t.Any], t.Any, t.Any]:
        yield from self._mapping.items()

    def keys(self):
        return t.cast(ConfigKeysView, list(self._mapping.keys()) + list(self._extra_content.keys()))

    def values(self):
        return t.cast(ConfigValuesView, [self[k] for k in self._mapping.keys()] + list(self._extra_content.values()))

    def items(self):
        return t.cast(
            ConfigItemsView, [(k, self[k]) for k in self._mapping.keys()] + list(self._extra_content.items())
        )

    def __iter__(self):
        return iter(list(self._mapping.keys()) + list(self._extra_content.keys()))

    def __contains__(self, item: t.Any):
        return item in self._mapping or item in self._extra_content

    def register(self, key: str, value: t.Any):
        """Register a new configuration in this mapping."""
        if key in self._mapping.keys():
            raise ValueError(f"'{key}' is already used by a OpenLLM config, pick another name.")
        self._extra_content[key] = value


CONFIG_MAPPING: dict[str, type[openllm.LLMConfig]] = _LazyConfigMapping(CONFIG_MAPPING_NAMES)

# The below handle special alias when we call underscore to the name directly
# without processing camelcase first.
CONFIG_NAME_ALIASES: dict[str, str] = {
    "chat_glm": "chatglm",
    "stable_lm": "stablelm",
    "star_coder": "starcoder",
    "gpt_neo_x": "gpt_neox",
    "lla_ma": "llama",
}


class AutoConfig:
    def __init__(self, *_: t.Any, **__: t.Any):
        """This metaclass should be initialised via `AutoConfig.for_model`."""
        raise EnvironmentError(
            "Cannot instantiate AutoConfig directly. Please use `AutoConfig.for_model(model_name)` instead."
        )

    @classmethod
    def for_model(cls, model_name: str, **attrs: t.Any) -> openllm.LLMConfig:
        model_name = inflection.underscore(model_name)
        if model_name in CONFIG_MAPPING:
            return CONFIG_MAPPING[model_name].model_construct_env(**attrs)
        raise ValueError(
            f"Unrecognized configuration class for {model_name}. "
            f"Model name should be one of {', '.join(CONFIG_MAPPING.keys())}."
        )

    @classmethod
    def infer_class_from_name(cls, name: str) -> type[openllm.LLMConfig]:
        model_name = inflection.underscore(name)
        if model_name in CONFIG_NAME_ALIASES:
            model_name = CONFIG_NAME_ALIASES[model_name]
        if model_name in CONFIG_MAPPING:
            return CONFIG_MAPPING[model_name]
        raise ValueError(
            f"Unrecognized configuration class for {model_name}. "
            f"Model name should be one of {', '.join(CONFIG_MAPPING.keys())}."
        )
