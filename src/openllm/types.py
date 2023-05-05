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

import bentoml
from bentoml._internal.io_descriptors.base import OpenAPIResponse
from bentoml._internal.models.model import \
    ModelSignaturesType as ModelSignaturesType
from bentoml.types import ModelSignatureDict

from openllm.configuration_utils import LLMConfig
from openllm.runner_utils import LLMRunnable
from openllm.utils import LazyLoader


class InferenceConfig(t.TypedDict):
    generate: ModelSignatureDict


class LLMModuleType(LazyLoader):
    @staticmethod
    def import_model(
        model_name: str,
        model_kwargs: dict[str, t.Any] | None = None,
        tokenizer_kwargs: dict[str, t.Any] | None = None,
        config_kwargs: dict[str, t.Any] | None = None,
    ) -> bentoml.Model:
        ...

    class LLMConfigImpl(LLMConfig):
        ...

    class LLMRunnableImpl(LLMRunnable[t.Any, t.Any], start_model_name="dummy"):
        ...

    @staticmethod
    def RunnableNameTokenizer(model_name: str | None = None, embedded: bool = True, **kwargs: t.Any) -> TokenizerRunner:
        ...


# The following type definition are extensions of bentoml.Runner
class TokenizerRunner(bentoml.Runner):
    ...
