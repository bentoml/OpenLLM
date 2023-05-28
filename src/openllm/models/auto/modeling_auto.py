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

from collections import OrderedDict

from .configuration_auto import CONFIG_MAPPING_NAMES
from .factory import _BaseAutoLLMClass, _LazyAutoMapping

MODEL_MAPPING_NAMES = OrderedDict(
    [
        ("flan_t5", "FlanT5"),
        ("dolly_v2", "DollyV2"),
        ("chatglm", "ChatGLM"),
        ("starcoder", "StarCoder"),
        ("falcon", "Falcon"),
        ("stablelm", "StableLM"),
    ]
)

MODEL_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, MODEL_MAPPING_NAMES)


class AutoLLM(_BaseAutoLLMClass):
    _model_mapping = MODEL_MAPPING
