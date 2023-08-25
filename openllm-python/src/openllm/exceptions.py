'''Base exceptions for OpenLLM. This extends BentoML exceptions.'''
from __future__ import annotations

from openllm_core.exceptions import Error as Error, FineTuneStrategyNotSupportedError as FineTuneStrategyNotSupportedError, ForbiddenAttributeError as ForbiddenAttributeError, GpuNotAvailableError as GpuNotAvailableError, MissingAnnotationAttributeError as MissingAnnotationAttributeError, MissingDependencyError as MissingDependencyError, OpenLLMException as OpenLLMException, ValidationError as ValidationError
