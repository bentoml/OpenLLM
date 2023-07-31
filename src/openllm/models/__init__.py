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
import sys
import typing as t
from ..utils import LazyModule

# fmt: off
# update-models-import.py: start module
_MODELS: set[str] = {'auto', 'baichuan', 'chatglm', 'dolly_v2', 'falcon', 'flan_t5', 'gpt_neox', 'llama', 'mpt', 'opt', 'stablelm', 'starcoder'}
# update-models-import.py: stop module
# fmt: on

if t.TYPE_CHECKING:
    # fmt: off
    # update-models-import.py: start types
    from . import auto as auto
    from . import baichuan as baichuan
    from . import chatglm as chatglm
    from . import dolly_v2 as dolly_v2
    from . import falcon as falcon
    from . import flan_t5 as flan_t5
    from . import gpt_neox as gpt_neox
    from . import llama as llama
    from . import mpt as mpt
    from . import opt as opt
    from . import stablelm as stablelm
    from . import starcoder as starcoder
    # update-models-import.py: stop types
    # fmt: on
else: sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], {k: [] for k in _MODELS}, module_spec=__spec__)
