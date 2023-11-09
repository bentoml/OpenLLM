from __future__ import annotations
import typing as t

import attr


@attr.define
class AgentRequest:
  inputs: str
  parameters: t.Dict[str, t.Any]


@attr.define
class AgentResponse:
  generated_text: str


@attr.define
class HFErrorResponse:
  error_code: int
  message: str
