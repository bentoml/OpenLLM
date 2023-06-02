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
"""
Utilities function for OpenLLM. User can import these function for convenience, but
we won't ensure backward compatibility for these functions. So use with caution.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import itertools
import logging
import os
import sys
import types
import typing as t

import attr
import inflection
from bentoml._internal.types import LazyType as LazyType
from bentoml._internal.types import get_origin as get_origin

# NOTE: The following exports useful utils from bentoml
from bentoml._internal.utils import LazyLoader as LazyLoader
from bentoml._internal.utils import bentoml_cattr as bentoml_cattr
from bentoml._internal.utils import copy_file_to_fs_folder as copy_file_to_fs_folder
from bentoml._internal.utils import pkg as pkg
from bentoml._internal.utils import reserve_free_port as reserve_free_port
from bentoml._internal.utils import resolve_user_filepath as resolve_user_filepath

from . import analytics as analytics
from . import codegen as codegen
from . import dantic as dantic
from .import_utils import ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES
from .import_utils import DummyMetaclass as DummyMetaclass
from .import_utils import is_cpm_kernels_available as is_cpm_kernels_available
from .import_utils import is_einops_available as is_einops_available
from .import_utils import is_flax_available as is_flax_available
from .import_utils import is_tf_available as is_tf_available
from .import_utils import is_torch_available as is_torch_available
from .import_utils import require_backends as require_backends

logger = logging.getLogger(__name__)

try:
    from typing import GenericAlias as TypingGenericAlias  # type: ignore
except ImportError:
    # python < 3.9 does not have GenericAlias (list[int], tuple[str, ...] and so on)
    TypingGenericAlias = ()

if sys.version_info < (3, 10):
    WithArgsTypes = (TypingGenericAlias,)
else:
    WithArgsTypes: t.Any = (
        t._GenericAlias,  # type: ignore (_GenericAlias is the actual GenericAlias implementation)
        types.GenericAlias,
        types.UnionType,
    )


def lenient_issubclass(cls: t.Any, class_or_tuple: type[t.Any] | tuple[type[t.Any], ...] | None) -> bool:
    try:
        return isinstance(cls, type) and issubclass(cls, class_or_tuple)  # type: ignore[arg-type]
    except TypeError:
        if isinstance(cls, WithArgsTypes):
            return False
        raise


@attr.define
class ModelEnv:
    model_name: str = attr.field(converter=inflection.underscore)

    @property
    def framework(self) -> str:
        return f"OPENLLM_{self.model_name.upper()}_FRAMEWORK"

    @property
    def model_config(self) -> str:
        return f"OPENLLM_{self.model_name.upper()}_CONFIG"

    @property
    def pretrained(self) -> str:
        return f"OPENLLM_{self.model_name.upper()}_PRETRAINED"

    @property
    def bettertransformer(self) -> str:
        return f"OPENLLM_{self.model_name.upper()}_BETTERTRANSFORMER"

    def gen_env_key(self, key: str) -> str:
        return f"OPENLLM_{self.model_name.upper()}_{key.upper()}"

    def convert_to_bettertransformer(self) -> bool:
        return os.environ.get(self.bettertransformer, str(False)).lower() == "true"

    @property
    def start_docstring(self) -> str:
        return getattr(self.module, f"START_{self.model_name.upper()}_COMMAND_DOCSTRING")

    @property
    def module(self) -> LazyLoader:
        return LazyLoader(self.model_name, globals(), f"openllm.models.{self.model_name}")

    def get_framework_env(self) -> t.Literal["pt", "flax", "tf"]:
        envvar = os.environ.get(self.framework, "pt")
        if envvar not in ("pt", "tf", "flax"):
            raise ValueError(f"Invalid framework implementation {envvar}, must be one of 'pt', 'tf', 'flax'")
        return envvar


class LazyModule(types.ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    This is a direct port from transformers.utils.import_utils._LazyModule for
    backwards compatibility with transformers < 4.18

    This is an extension a more powerful LazyLoader.
    """

    # Very heavily inspired by optuna.integration._IntegrationModule
    # https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: dict[str, list[str]],
        module_spec: importlib.machinery.ModuleSpec | None = None,
        extra_objects: dict[str, t.Any] | None = None,
    ):
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module: dict[str, str] = {}
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(itertools.chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = {} if extra_objects is None else extra_objects
        self._name = name
        self._import_structure = import_structure

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = t.cast("list[str]", super().__dir__())
        # The elements of self.__all__ that are submodules may or
        # may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the
        # elements of self.__all__ that are not already in the dir.
        for attribute in self.__all__:
            if attribute not in result:
                result.append(attribute)
        return result

    def __getattr__(self, name: str) -> t.Any:
        if name in self._objects:
            return self._objects[name]
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module[name])
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str):
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (self.__class__, (self._name, self.__file__, self._import_structure))
