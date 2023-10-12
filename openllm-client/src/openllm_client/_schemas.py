from __future__ import annotations
import typing as t

import attr
import cattr

class _Mixin:
  def json(self) -> dict[str, t.Any]:
    if not attr.has(self.__class__): raise TypeError(f'Class {self.__class__} must be attr class')
    return cattr.unstructure(self)

@attr.define
class Request(_Mixin):
  prompt: str
  llm_config: t.Dict[str, t.Any]

@attr.define
class Response(_Mixin):
  responses: t.List[str]
  configuration: t.Dict[str, t.Any]

@attr.define
class StreamResponse(_Mixin):
  text: str
