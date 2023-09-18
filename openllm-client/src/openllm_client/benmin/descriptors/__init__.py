"""'openllm_client.benmin.decsriptors' holds a minimal implementation of some IO Descriptors protocol.

This is subjected to change as we are currently refactoring and rework on the new generation of IO descriptors protocol. This package is meant to be used internally
and vendorred for performance reasons."""
from __future__ import annotations
import typing as t

_supported: dict[str, t.Any] = {'bentoml.io.JSON': 'json', 'bentoml.io.Text': 'text'}

class OpenApiSpec(t.TypedDict):
  id: str
  args: t.Dict[str, t.Any]

def from_spec(spec: dict[str, t.Any]) -> IODescriptor[t.Any]:
  ...

T = t.TypeVar('T')

class IODescriptor(t.Generic[T]):
  ...
