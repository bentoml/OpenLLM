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
from ...utils import is_torch_available


_import_structure: dict[str, list[str]] = {
    "configuration_mpt": ["MPTConfig", "START_MPT_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE", "PROMPT_MAPPING"],
}

try:
    if not is_torch_available():
        raise MissingDependencyError
except MissingDependencyError:
    pass
else:
    _import_structure["modeling_mpt"] = ["MPT"]


if t.TYPE_CHECKING:
    from .configuration_mpt import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
    from .configuration_mpt import PROMPT_MAPPING as PROMPT_MAPPING
    from .configuration_mpt import START_MPT_COMMAND_DOCSTRING as START_MPT_COMMAND_DOCSTRING
    from .configuration_mpt import MPTConfig as MPTConfig

    try:
        if not is_torch_available():
            raise MissingDependencyError
    except MissingDependencyError:
        pass
    else:
        from .modeling_mpt import MPT as MPT
else:
    import sys

    sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
