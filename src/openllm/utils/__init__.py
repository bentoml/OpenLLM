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
import os
import sys
import types
import typing as t

from bentoml._internal.configuration import get_debug_mode as get_debug_mode
from bentoml._internal.configuration import get_quiet_mode as get_quiet_mode
from bentoml._internal.configuration import set_debug_mode as set_debug_mode
from bentoml._internal.configuration import set_quiet_mode as set_quiet_mode
from bentoml._internal.log import configure_logging as configure_logging
from bentoml._internal.log import \
    configure_server_logging as configure_server_logging
from bentoml._internal.types import LazyType as LazyType
# NOTE: The following exports useful utils from bentoml
from bentoml._internal.utils import LazyLoader as LazyLoader
from bentoml._internal.utils import bentoml_cattr as bentoml_cattr
from bentoml._internal.utils import \
    copy_file_to_fs_folder as copy_file_to_fs_folder
from bentoml._internal.utils import first_not_none as first_not_none
from bentoml._internal.utils import pkg as pkg
from bentoml._internal.utils import reserve_free_port as reserve_free_port
from bentoml._internal.utils import \
    resolve_user_filepath as resolve_user_filepath

from .lazy import LazyModule

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


# XXX: define all classes, functions import above this line
# since _extras will be the locals() import from this file.
_extras: dict[str, t.Any] = {
    k: v for k, v in locals().items() if not isinstance(v, types.ModuleType) and not k.startswith("_")
}

_import_structure = {
    "analytics": [],
    "codegen": [],
    "dantic": [],
    "import_utils": [
        "ENV_VARS_TRUE_VALUES",
        "DummyMetaclass",
        "ModelEnv",
        "is_cpm_kernels_available",
        "is_einops_available",
        "is_flax_available",
        "is_tf_available",
        "is_torch_available",
        "is_bitsandbytes_available",
        "require_backends",
    ],
}

if t.TYPE_CHECKING:
    from . import analytics as analytics
    from . import codegen as codegen
    from . import dantic as dantic
    from .import_utils import ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES
    from .import_utils import DummyMetaclass as DummyMetaclass
    from .import_utils import ModelEnv as ModelEnv
    from .import_utils import \
        is_bitsandbytes_available as is_bitsandbytes_available
    from .import_utils import \
        is_cpm_kernels_available as is_cpm_kernels_available
    from .import_utils import is_einops_available as is_einops_available
    from .import_utils import is_flax_available as is_flax_available
    from .import_utils import is_tf_available as is_tf_available
    from .import_utils import is_torch_available as is_torch_available
    from .import_utils import require_backends as require_backends
else:
    import sys

    sys.modules[__name__] = LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extras,
    )
