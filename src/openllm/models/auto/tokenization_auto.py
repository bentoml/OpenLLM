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
from collections import OrderedDict

import openllm

from .configuration_auto import _LazyConfigMapping

TOKENIZER_MAPPING_NAMES = OrderedDict([("flan_t5", "FlanT5Tokenizer")])

TOKENIZER_MAPPING = _LazyConfigMapping(TOKENIZER_MAPPING_NAMES)


class AutoTokenizer:
    def __init__(self):
        raise EnvironmentError(
            "This class should not be initialized directly. Instead use 'Tokenizer.create_runner' instead"
        )

    @classmethod
    def create_runner(cls, model_name: str, pretrained_or_path: str | None = None, **kwargs: t.Any):
        model_name = openllm.utils.kebab_to_snake_case(model_name)
        if model_name in TOKENIZER_MAPPING:
            tokenizer_class = TOKENIZER_MAPPING[model_name]
            if pretrained_or_path is None:
                pretrained_or_path = openllm.utils.get_pretrained_env(tokenizer_class.start_model_name)
            return tokenizer_class(pretrained_or_path, **kwargs)
        raise ValueError(
            f"Unrecognized model {model_name} to build an Tokenizer.\n"
            f"Model type should be one of {', '.join(TOKENIZER_MAPPING.keys())}."
        )
