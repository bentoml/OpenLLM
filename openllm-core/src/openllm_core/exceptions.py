"""Base exceptions for OpenLLM. This extends BentoML exceptions."""

from __future__ import annotations
from http import HTTPStatus


class OpenLLMException(Exception):
  """Base class for all OpenLLM exceptions. This shares similar interface with BentoMLException."""

  error_code = HTTPStatus.INTERNAL_SERVER_ERROR

  def __init__(self, message: str):
    self.message = message
    super().__init__(message)


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


class Error(BaseException):
  """To be used instead of naked raise."""


class FineTuneStrategyNotSupportedError(OpenLLMException):
  """Raised when a fine-tune strategy is not supported for given LLM."""


class ModelNotFound(OpenLLMException):
  """Raised when a model is not found for given model_id."""

  error_code = HTTPStatus.NOT_FOUND
