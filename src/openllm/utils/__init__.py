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

import sys
import types
import typing as t

from bentoml._internal.types import LazyType

# NOTE: The following exports useful utils from bentoml
from bentoml._internal.utils import (
    LazyLoader,
    bentoml_cattr,
    copy_file_to_fs_folder,
    first_not_none,
    pkg,
    reserve_free_port,
    resolve_user_filepath,
)

from .lazy import LazyModule

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
    from .import_utils import is_cpm_kernels_available as is_cpm_kernels_available
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
        extra_objects={
            "pkg": pkg,
            "LazyModule": LazyModule,
            "LazyType": LazyType,
            "LazyLoader": LazyLoader,
            "bentoml_cattr": bentoml_cattr,
            "copy_file_to_fs_folder": copy_file_to_fs_folder,
            "first_not_none": first_not_none,
            "reserve_free_port": reserve_free_port,
            "lenient_issubclass": lenient_issubclass,
            "resolve_user_filepath": resolve_user_filepath,
        },
    )
