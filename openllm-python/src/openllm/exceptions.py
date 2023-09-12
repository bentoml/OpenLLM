'''Base exceptions for OpenLLM. This extends BentoML exceptions.'''
from __future__ import annotations

from openllm_core.exceptions import Error as Error
from openllm_core.exceptions import FineTuneStrategyNotSupportedError as FineTuneStrategyNotSupportedError
from openllm_core.exceptions import ForbiddenAttributeError as ForbiddenAttributeError
from openllm_core.exceptions import GpuNotAvailableError as GpuNotAvailableError
from openllm_core.exceptions import MissingAnnotationAttributeError as MissingAnnotationAttributeError
from openllm_core.exceptions import MissingDependencyError as MissingDependencyError
from openllm_core.exceptions import OpenLLMException as OpenLLMException
from openllm_core.exceptions import ValidationError as ValidationError
