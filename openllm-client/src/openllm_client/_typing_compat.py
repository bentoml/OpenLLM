from typing import Literal

from openllm_core._typing_compat import (
  Annotated as Annotated,
  LiteralString as LiteralString,
  NotRequired as NotRequired,
  Required as Required,
  Self as Self,
  TypeGuard as TypeGuard,
  TypedDict as TypedDict,
  dataclass_transform as dataclass_transform,
  overload as overload,
)

Platform = Annotated[
  LiteralString, Literal['MacOS', 'Linux', 'Windows', 'FreeBSD', 'OpenBSD', 'iOS', 'iPadOS', 'Android', 'Unknown'], str
]
Architecture = Annotated[LiteralString, Literal['arm', 'arm64', 'x86', 'x86_64', 'Unknown'], str]
