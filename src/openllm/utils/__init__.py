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
from __future__ import annotations as _annotations

import functools
import logging
import logging.config
import os
import sys
import types
import typing as t

from bentoml._internal.configuration import get_debug_mode
from bentoml._internal.configuration import get_quiet_mode
from bentoml._internal.configuration import set_debug_mode
from bentoml._internal.configuration import set_quiet_mode
from bentoml._internal.log import CLI_LOGGING_CONFIG as _CLI_LOGGING_CONFIG
from bentoml._internal.types import LazyType
from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils import bentoml_cattr
from bentoml._internal.utils import copy_file_to_fs_folder
from bentoml._internal.utils import first_not_none
from bentoml._internal.utils import pkg
from bentoml._internal.utils import reserve_free_port
from bentoml._internal.utils import resolve_user_filepath

from .lazy import LazyModule


logger = logging.getLogger(__name__)

try:
    from typing import GenericAlias as _TypingGenericAlias  # type: ignore
except ImportError:
    # python < 3.9 does not have GenericAlias (list[int], tuple[str, ...] and so on)
    _TypingGenericAlias = ()

if sys.version_info < (3, 10):
    _WithArgsTypes = (_TypingGenericAlias,)
else:
    _WithArgsTypes: t.Any = (
        t._GenericAlias,  # type: ignore (_GenericAlias is the actual GenericAlias implementation)
        types.GenericAlias,
        types.UnionType,
    )


def lenient_issubclass(cls: t.Any, class_or_tuple: type[t.Any] | tuple[type[t.Any], ...] | None) -> bool:
    try:
        return isinstance(cls, type) and issubclass(cls, class_or_tuple)  # type: ignore[arg-type]
    except TypeError:
        if isinstance(cls, _WithArgsTypes):
            return False
        raise


def gpu_count() -> tuple[int, ...]:
    from bentoml._internal.resource import NvidiaGpuResource

    return tuple(NvidiaGpuResource.from_system())


# equivocal setattr to save one lookup per assignment
_object_setattr = object.__setattr__


def non_intrusive_setattr(obj: t.Any, name: str, value: t.Any) -> None:
    """This makes sure that we don't overwrite any existing attributes on the object"""
    _setattr = functools.partial(setattr, obj) if isinstance(obj, type) else _object_setattr.__get__(obj)

    if not hasattr(obj, name):
        _setattr(name, value)


DEBUG = sys.flags.dev_mode or (not sys.flags.ignore_environment and bool(os.environ.get("OPENLLMDEVDEBUG")))


_LOGGING_CONFIG = _CLI_LOGGING_CONFIG.copy()
_LOGGING_CONFIG["loggers"].update(
    {
        "openllm": {
            "level": logging.INFO,
            "handlers": ["bentomlhandler", "defaulthandler"],
            "propagate": False,
        }
    }
)


def configure_logging() -> None:
    """Configure logging for OpenLLM. Behaves similar to how BentoML loggers
    are being configured."""
    if get_quiet_mode():
        _LOGGING_CONFIG["loggers"]["openllm"]["level"] = logging.ERROR
        _LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.ERROR
        _LOGGING_CONFIG["root"]["level"] = logging.ERROR
    elif get_debug_mode() or DEBUG:
        _LOGGING_CONFIG["loggers"]["openllm"]["level"] = logging.DEBUG
        _LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.DEBUG
        _LOGGING_CONFIG["root"]["level"] = logging.DEBUG
    else:
        _LOGGING_CONFIG["loggers"]["openllm"]["level"] = logging.INFO
        _LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.INFO
        _LOGGING_CONFIG["root"]["level"] = logging.INFO

    logging.config.dictConfig(_LOGGING_CONFIG)


# NOTE: The set marks contains a set of modules name
# that are available above and are whitelisted
# to be included in the extra_objects map.
_whitelist_modules = {"pkg"}

# XXX: define all classes, functions import above this line
# since _extras will be the locals() import from this file.
_extras: dict[str, t.Any] = {
    k: v
    for k, v in locals().items()
    if k in _whitelist_modules or (not isinstance(v, types.ModuleType) and not k.startswith("_"))
}

_extras["__openllm_migration__"] = {"ModelEnv": "EnvVarMixin"}

_import_structure = {
    "analytics": [],
    "codegen": [],
    "dantic": [],
    "constants": [],
    "representation": ["ReprMixin"],
    "import_utils": [
        "OPTIONAL_DEPENDENCIES",
        "ENV_VARS_TRUE_VALUES",
        "DummyMetaclass",
        "EnvVarMixin",
        "requires_dependencies",
        "is_cpm_kernels_available",
        "is_einops_available",
        "is_flax_available",
        "is_tf_available",
        "is_torch_available",
        "is_bitsandbytes_available",
        "is_peft_available",
        "is_datasets_available",
        "is_transformers_supports_kbit",
        "is_transformers_supports_agent",
        "require_backends",
    ],
}

if t.TYPE_CHECKING:
    # NOTE: The following exports useful utils from bentoml
    from . import LazyLoader as LazyLoader
    from . import LazyType as LazyType
    from . import analytics as analytics
    from . import bentoml_cattr as bentoml_cattr
    from . import codegen as codegen
    from . import configure_logging as configure_logging
    from . import constants as constants
    from . import copy_file_to_fs_folder as copy_file_to_fs_folder
    from . import dantic as dantic
    from . import first_not_none as first_not_none
    from . import get_debug_mode as get_debug_mode
    from . import get_quiet_mode as get_quiet_mode
    from . import gpu_count as gpu_count
    from . import lenient_issubclass as lenient_issubclass
    from . import non_intrusive_setattr as non_intrusive_setattr
    from . import pkg as pkg
    from . import reserve_free_port as reserve_free_port
    from . import resolve_user_filepath as resolve_user_filepath
    from . import set_debug_mode as set_debug_mode
    from . import set_quiet_mode as set_quiet_mode
    from .import_utils import ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES
    from .import_utils import OPTIONAL_DEPENDENCIES as OPTIONAL_DEPENDENCIES
    from .import_utils import DummyMetaclass as DummyMetaclass
    from .import_utils import EnvVarMixin as EnvVarMixin
    from .import_utils import is_bitsandbytes_available as is_bitsandbytes_available
    from .import_utils import is_cpm_kernels_available as is_cpm_kernels_available
    from .import_utils import is_datasets_available as is_datasets_available
    from .import_utils import is_einops_available as is_einops_available
    from .import_utils import is_flax_available as is_flax_available
    from .import_utils import is_peft_available as is_peft_available
    from .import_utils import is_tf_available as is_tf_available
    from .import_utils import is_torch_available as is_torch_available
    from .import_utils import is_transformers_supports_agent as is_transformers_supports_agent
    from .import_utils import is_transformers_supports_kbit as is_transformers_supports_kbit
    from .import_utils import require_backends as require_backends
    from .import_utils import requires_dependencies as requires_dependencies
    from .lazy import LazyModule as LazyModule
    from .representation import ReprMixin as ReprMixin
else:
    import sys

    sys.modules[__name__] = LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extras,
    )
