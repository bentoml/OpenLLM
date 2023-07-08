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
"""Base exceptions for OpenLLM. This extends BentoML exceptions."""
from __future__ import annotations

import bentoml


class OpenLLMException(bentoml.exceptions.BentoMLException):
    """Base class for all OpenLLM exceptions. This extends BentoMLException."""


class GpuNotAvailableError(OpenLLMException):
    """Raised when there is no GPU available in given system."""


class ValidationError(OpenLLMException):
    """Raised when a validation fails."""


class ForbiddenAttributeError(OpenLLMException):
    """Raised when using an _internal field."""


class MissingAnnotationAttributeError(OpenLLMException):
    """Raised when a field under openllm.LLMConfig is missing annotations."""


class MissingDependencyError(BaseException):
    """Raised when a dependency is missing."""


class FineTuneStrategyNotSupportedError(OpenLLMException):
    """Raised when a fine-tune strategy is not supported for given LLM."""
