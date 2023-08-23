'''Base exceptions for OpenLLM. This extends BentoML exceptions.'''
from __future__ import annotations
from openllm_core.exceptions import OpenLLMException as OpenLLMException, GpuNotAvailableError as GpuNotAvailableError, ValidationError as ValidationError, ForbiddenAttributeError as ForbiddenAttributeError, MissingAnnotationAttributeError as MissingAnnotationAttributeError, MissingDependencyError as MissingDependencyError, Error as Error, FineTuneStrategyNotSupportedError as FineTuneStrategyNotSupportedError
