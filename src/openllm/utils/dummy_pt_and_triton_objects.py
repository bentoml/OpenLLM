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

from ..utils import DummyMetaclass
from ..utils import require_backends


class MPT(metaclass=DummyMetaclass):
    _backends = ["torch", "triton"]

    def __init__(self, *args: t.Any, **attrs: t.Any):
        require_backends(self, ["torch"])
