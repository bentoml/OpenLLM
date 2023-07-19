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
import importlib.machinery
import itertools
import os
import types
import typing as t
import warnings

from ..exceptions import ForbiddenAttributeError
from ..exceptions import OpenLLMException


class UsageNotAllowedError(OpenLLMException):
    """Raised when LazyModule.__getitem__ is forbidden."""


class MissingAttributesError(OpenLLMException):
    """Raised when given keys is not available in LazyModule special mapping."""


_sentinel = object()

_reserved_namespace = {"__openllm_special__", "__openllm_migration__"}


class LazyModule(types.ModuleType):
    """Module class that surfaces all objects but only performs associated imports when the objects are requested.

    This is a direct port from transformers.utils.import_utils._LazyModule for backwards compatibility with transformers < 4.18.

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
        doc: str | None = None,
        extra_objects: dict[str, t.Any] | None = None,
    ):
        """Lazily load this module as an object.

        It does instantiate a __all__ and __dir__ for IDE support

        Args:
            name: module name
            module_file: the given file. Often default to 'globals()['__file__']'
            import_structure: A dictionary of module and its corresponding attributes that can be loaded from given 'module'
            module_spec: __spec__ of the lazily loaded module
            doc: Optional docstring for this module.
            extra_objects: Any additional objects that this module can also be accessed. Useful for additional metadata as well
                           as any locals() functions
        """
        super().__init__(name)
        self._modules = set(import_structure.keys())
        self._class_to_module: dict[str, str] = {}
        _extra_objects = {} if extra_objects is None else extra_objects
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = list(import_structure.keys()) + list(itertools.chain(*import_structure.values()))
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self.__doc__ = doc
        self._objects = _extra_objects
        self._name = name
        self._import_structure = import_structure

    def __dir__(self) -> list[str]:
        """Needed for autocompletion in an IDE."""
        result = t.cast("list[str]", super().__dir__())
        # The elements of self.__all__ that are submodules may or
        # may not be in the dir already, depending on whether
        # they have been accessed or not. So we only add the
        # elements of self.__all__ that are not already in the dir.
        return result + [i for i in self.__all__ if i not in result]

    def __getitem__(self, key: str) -> t.Any:
        """This is reserved to only internal uses and users shouldn't use this."""
        if self._objects.get("__openllm_special__") is None:
            raise UsageNotAllowedError(f"'{self._name}' is not allowed to be used as a dict.")
        _special_mapping = self._objects.get("__openllm_special__", {})
        try:
            if key in _special_mapping:
                return getattr(self, _special_mapping.__getitem__(key))
            raise MissingAttributesError(f"Requested '{key}' is not available in given mapping.")
        except AttributeError as e:
            raise KeyError(f"'{self._name}' has no attribute {_special_mapping[key]}") from e
        except Exception as e:
            raise KeyError(f"Failed to lookup '{key}' in '{self._name}'") from e

    def __getattr__(self, name: str) -> t.Any:
        """Equivocal __getattr__ implementation.

        It checks from _objects > _modules and does it recursively.
        """
        if name in _reserved_namespace:
            raise ForbiddenAttributeError(
                f"'{name}' is a reserved namespace for {self._name} and should not be access nor modified."
            )
        if "__openllm_migration__" in self._objects:
            cur_value = self._objects["__openllm_migration__"].get(name, _sentinel)
            if cur_value is not _sentinel:
                warnings.warn(
                    f"'{name}' is deprecated and will be removed in future version. Make sure to use '{cur_value}' instead",
                    DeprecationWarning,
                    stacklevel=3,
                )
                return getattr(self, cur_value)
        if name in self._objects:
            return self._objects.__getitem__(name)
        if name in self._modules:
            value = self._get_module(name)
        elif name in self._class_to_module.keys():
            module = self._get_module(self._class_to_module.__getitem__(name))
            value = getattr(module, name)
        else:
            raise AttributeError(f"module {self.__name__} has no attribute {name}")

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> types.ModuleType:
        try:
            return importlib.import_module("." + module_name, self.__name__)
        except Exception as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self) -> tuple[type[LazyModule], tuple[str, str | None, dict[str, list[str]]]]:
        """This is to ensure any given module is pickle-able."""
        return (self.__class__, (self._name, self.__file__, self._import_structure))
