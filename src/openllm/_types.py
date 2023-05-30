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
Types definition for OpenLLM.

Note that this module SHOULD NOT BE IMPORTED DURING RUNTIME, as this serve only for typing purposes.
"""
from __future__ import annotations

import typing as t

if not t.TYPE_CHECKING:
    raise RuntimeError(f"{__name__} should not be imported during runtime")

import transformers
from bentoml._internal.models.model import ModelSignatureDict as ModelSignatureDict
from bentoml._internal.models.model import ModelSignaturesType as ModelSignaturesType

LLMModel = transformers.PreTrainedModel | transformers.TFPreTrainedModel | transformers.FlaxPreTrainedModel
LLMTokenizer = (
    transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast | transformers.PreTrainedTokenizerBase
)
