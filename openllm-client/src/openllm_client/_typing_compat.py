from typing import Literal

from openllm_core._typing_compat import Annotated as Annotated
from openllm_core._typing_compat import LiteralString as LiteralString
from openllm_core._typing_compat import NotRequired as NotRequired
from openllm_core._typing_compat import Required as Required
from openllm_core._typing_compat import Self as Self
from openllm_core._typing_compat import dataclass_transform as dataclass_transform
from openllm_core._typing_compat import overload as overload


Platform = Annotated[
  LiteralString, Literal['MacOS', 'Linux', 'Windows', 'FreeBSD', 'OpenBSD', 'iOS', 'iPadOS', 'Android', 'Unknown'], str
]
Architecture = Annotated[LiteralString, Literal['arm', 'arm64', 'x86', 'x86_64', 'Unknown'], str]
