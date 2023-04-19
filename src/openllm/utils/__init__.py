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
import re
import types
import typing as t

import bentoml
# NOTE: The following exports useful utils from bentoml
from bentoml._internal.utils import LazyLoader as LazyLoader
from bentoml._internal.utils import pkg as packaging_utils
from bentoml._internal.utils import reserve_free_port as reserve_free_port
from bentoml._internal.utils import \
    resolve_user_filepath as resolve_user_filepath

if t.TYPE_CHECKING:
    import transformers
    import transformers.utils as import_utils_shim

    from openllm.runner_utils import LLMRunner

else:
    transformers = LazyLoader("transformers", globals(), "transformers")

    if packaging_utils.pkg_version_info("transformers")[:2] >= (4, 18):
        import_utils_shim = LazyLoader("import_utils_shim", globals(), "transformers.utils")
    else:
        # NOTE: This logic below handle 4.18 compatibility for checking tf, jax, and torch availability.
        import_utils_shim = LazyLoader(
            "import_utils_shim", globals(), "bentoml._internal.frameworks.utils.transformers"
        )

logger = logging.getLogger(__name__)

_object_setattr = object.__setattr__


def kebab_to_snake_case(name: str) -> str:
    """Convert a given kebab-case name to snake_case"""
    return re.sub(r"-", "_", name)


def get_pretrained_env(model_name: str) -> str | None:
    """Convert a given runnable start model name (kebab-case) to a
    ENV variable snake_case (OPENLLM_FLAN_T5_PRETRAINED)"""
    return os.environ.get(f"OPENLLM_{kebab_to_snake_case(model_name.upper())}_PRETRAINED", None)


def get_lazy_module(model_name: str) -> LazyLoader:
    snaked_model_name = kebab_to_snake_case(model_name)
    return LazyLoader(snaked_model_name, globals(), f"openllm.models.{snaked_model_name}")


def get_working_dir(model_name: str) -> str:
    """Get the working directory for a given model name"""
    return os.path.dirname(importlib.import_module(f".{kebab_to_snake_case(model_name)}", "openllm.models").__file__)


def FRAMEWORK_ENV_VAR(model_name: str) -> str:
    return f"OPENLLM_{kebab_to_snake_case(model_name.upper())}_FRAMEWORK"


def get_framework_env(model_name: str) -> str:
    envvar = os.environ.get(FRAMEWORK_ENV_VAR(model_name), "pt")
    if envvar not in ("pt", "tf", "flax"):
        raise ValueError(f"Invalid framework implementation {envvar}, must be one of 'pt', 'tf', 'flax'")
    return envvar


def generate_service_name(runner: LLMRunner) -> str:
    dummy_object = runner.runnable_class.dummy_object()
    return f"llm-{dummy_object.start_model_name}-service"


def convert_transformers_model_name(name: str) -> str:
    return re.sub("[^a-zA-Z0-9]+", "-", name)


def generate_tag_from_model_name(model_name: str, prefix: str | None = None, **kwargs: t.Any) -> bentoml.Tag:
    """Generate a ``bentoml.Tag`` from a given transformers model name.

    Note that this depends on your model to have a config class available.

    Args:
        model_name: The transformers model name.
        **kwargs: Additional kwargs to pass to the ``transformers.AutoConfig`` constructor.
                  If your pass ``return_unused_kwargs=True``, it will be ignored.
    """
    if "return_unused_kwargs" in kwargs:
        logger.debug("Ignoring 'return_unused_kwargs' in 'generate_tag_from_model_name'.")
        kwargs.pop("return_unused_kwargs")
    config = transformers.AutoConfig.from_pretrained(model_name, **kwargs)
    commit_hash = getattr(config, "_commit_hash", None)
    if commit_hash is None:
        logger.warning(
            "Given %s from '%s' doesn't contain a commit hash. We will generate the tag without specific version.",
            config.__class__,
            model_name,
        )
    tag_str = (
        convert_transformers_model_name(model_name)
        if commit_hash is None
        else f"{convert_transformers_model_name(model_name)}:{commit_hash}"
    )
    return bentoml.Tag.from_taglike((f"{prefix}-" if prefix is not None else "") + tag_str)


class LazyModule(types.ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    This is a direct port from transformers.utils.import_utils._LazyModule for backwards compatibility with transformers <4.18

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
        for attr in self.__all__:
            if attr not in result:
                result.append(attr)
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
