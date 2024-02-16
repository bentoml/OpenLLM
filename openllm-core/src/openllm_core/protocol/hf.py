from __future__ import annotations
import pydantic, typing as t


class AgentRequest(pydantic.BaseModel):
  inputs: str
  parameters: t.Dict[str, t.Any]


class AgentResponse(pydantic.BaseModel):
  generated_text: str


class HFErrorResponse(pydantic.BaseModel):
  error_code: int
  message: str
