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

from ...exceptions import MissingDependencyError
from ...utils import LazyModule
from ...utils import is_flax_available
from ...utils import is_tf_available
from ...utils import is_torch_available


_import_structure: dict[str, list[str]] = {
    "configuration_flan_t5": ["FlanT5Config", "START_FLAN_T5_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"],
}

try:
    if not is_torch_available():
        raise MissingDependencyError
except MissingDependencyError:
    pass
else:
    _import_structure["modeling_flan_t5"] = ["FlanT5"]

try:
    if not is_flax_available():
        raise MissingDependencyError
except MissingDependencyError:
    pass
else:
    _import_structure["modeling_flax_flan_t5"] = ["FlaxFlanT5"]

try:
    if not is_tf_available():
        raise MissingDependencyError
except MissingDependencyError:
    pass
else:
    _import_structure["modeling_tf_flan_t5"] = ["TFFlanT5"]


if t.TYPE_CHECKING:
    from .configuration_flan_t5 import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
    from .configuration_flan_t5 import START_FLAN_T5_COMMAND_DOCSTRING as START_FLAN_T5_COMMAND_DOCSTRING
    from .configuration_flan_t5 import FlanT5Config as FlanT5Config

    try:
        if not is_torch_available():
            raise MissingDependencyError
    except MissingDependencyError:
        pass
    else:
        from .modeling_flan_t5 import FlanT5 as FlanT5

    try:
        if not is_flax_available():
            raise MissingDependencyError
    except MissingDependencyError:
        pass
    else:
        from .modeling_flax_flan_t5 import FlaxFlanT5 as FlaxFlanT5

    try:
        if not is_tf_available():
            raise MissingDependencyError
    except MissingDependencyError:
        pass
    else:
        from .modeling_tf_flan_t5 import TFFlanT5 as TFFlanT5
else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
