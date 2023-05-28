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
import inflection
from bentoml._internal.types import LazyType as LazyType

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
from .import_utils import DummyMetaclass as DummyMetaclass
from .import_utils import is_cpm_kernels_available as is_cpm_kernels_available
from .import_utils import is_einops_available as is_einops_available
from .import_utils import is_flax_available as is_flax_available
from .import_utils import is_tf_available as is_tf_available
from .import_utils import is_torch_available as is_torch_available
from .import_utils import require_backends as require_backends

if t.TYPE_CHECKING:
    import transformers
else:
    transformers = LazyLoader("transformers", globals(), "transformers")

logger = logging.getLogger(__name__)

_object_setattr = object.__setattr__


def get_lazy_module(model_name: str) -> LazyLoader:
    snaked_model_name = inflection.underscore(model_name)
    return LazyLoader(snaked_model_name, globals(), f"openllm.models.{snaked_model_name}")


def FRAMEWORK_ENV_VAR(model_name: str) -> str:
    return f"OPENLLM_{inflection.underscore(model_name).upper()}_FRAMEWORK"


def MODEL_CONFIG_ENV_VAR(model_name: str) -> str:
    return f"OPENLLM_{inflection.underscore(model_name).upper()}_CONFIG"


def get_framework_env(model_name: str) -> t.Literal["pt", "flax", "tf"]:
    envvar = os.environ.get(FRAMEWORK_ENV_VAR(model_name), "pt")
    if envvar not in ("pt", "tf", "flax"):
        raise ValueError(f"Invalid framework implementation {envvar}, must be one of 'pt', 'tf', 'flax'")
    return envvar


def convert_transformers_model_name(name: str) -> str:
    if os.path.exists(os.path.dirname(name)):
        name = os.path.basename(name)
        logger.debug("Given name is a path, only returning the basename %s")
        return name
    return re.sub("[^a-zA-Z0-9]+", "-", name)


def generate_tags(
    model_name_or_path: str, prefix: str | None = None, **attrs: t.Any
) -> tuple[bentoml.Tag, dict[str, t.Any]]:
    """Generate a ``bentoml.Tag`` from a given transformers model name.

    Note that this depends on your model to have a config class available.

    Args:
        model_name_or_path: The transformers model name or path to load the model from.
                            If it is a path, then `openllm_model_version` must be passed in as a kwarg.
        prefix: The prefix to prepend to the tag. If None, then no prefix will be prepended.
        **attrs: Additional kwargs to pass to the ``transformers.AutoConfig`` constructor.
                  If your pass ``return_unused_kwargs=True``, it will be ignored.

    Returns:
        A tuple of ``bentoml.Tag`` and a dict of unused kwargs.
    """
    if "return_unused_kwargs" in attrs:
        logger.debug("Ignoring 'return_unused_kwargs' in 'generate_tag_from_model_name'.")
        attrs.pop("return_unused_kwargs", None)

    config, attrs = t.cast(
        "tuple[transformers.PretrainedConfig, dict[str, t.Any]]",
        transformers.AutoConfig.from_pretrained(model_name_or_path, return_unused_kwargs=True, **attrs),
    )
    name = convert_transformers_model_name(model_name_or_path)

    if os.path.exists(os.path.dirname(model_name_or_path)):
        # If the model_name_or_path is a path, we assume it's a local path,
        # then users must pass a version for this.
        model_version = attrs.pop("openllm_model_version", None)
        if model_version is None:
            logger.warning(
                """\
        When passing a path, it is recommended to also pass 'openllm_model_version' into Runner/AutoLLM intialization.

        For example:

        >>> import openllm
        >>> runner = openllm.Runner('/path/to/fine-tuning/model', openllm_model_version='lora-version')

        Example with AutoLLM:

        >>> import openllm
        >>> model = openllm.AutoLLM.for_model('/path/to/fine-tuning/model', openllm_model_version='lora-version')

        No worries, OpenLLM will generate one for you. But for your own convenience, make sure to 
        specify 'openllm_model_version'.
        """
            )
            model_version = bentoml.Tag.from_taglike(name).make_new_version().version
    else:
        model_version = getattr(config, "_commit_hash", None)
        if model_version is None:
            logger.warning(
                "Given %s from '%s' doesn't contain a commit hash. We will generate the tag without specific version.",
                t.cast("type[transformers.PretrainedConfig]", config.__class__),
                model_name_or_path,
            )
    return bentoml.Tag.from_taglike((f"{prefix}-" if prefix is not None else "") + f"{name}:{model_version}"), attrs


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
