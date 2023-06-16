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
        _extra_objects = {} if extra_objects is None else extra_objects
        for key, values in import_structure.items():
            for value in values:
                self._class_to_module[value] = key
        # Needed for autocompletion in an IDE
        self.__all__ = (
            list(import_structure.keys()) + list(itertools.chain(*import_structure.values())) + list(_extra_objects)
        )
        self.__file__ = module_file
        self.__spec__ = module_spec
        self.__path__ = [os.path.dirname(module_file)]
        self._objects = _extra_objects
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
