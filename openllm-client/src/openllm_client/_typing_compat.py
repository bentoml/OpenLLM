import sys

from typing import Literal


if sys.version_info[:2] >= (3, 11):
  from typing import LiteralString as LiteralString
  from typing import NotRequired as NotRequired
  from typing import Required as Required
  from typing import Self as Self
  from typing import dataclass_transform as dataclass_transform
  from typing import overload as overload
else:
  from typing_extensions import LiteralString as LiteralString
  from typing_extensions import NotRequired as NotRequired
  from typing_extensions import Required as Required
  from typing_extensions import Self as Self
  from typing_extensions import dataclass_transform as dataclass_transform
  from typing_extensions import overload as overload

if sys.version_info[:2] >= (3, 9):
  from typing import Annotated as Annotated
else:
  from typing_extensions import Annotated as Annotated

Platform = Annotated[
  LiteralString, Literal['MacOS', 'Linux', 'Windows', 'FreeBSD', 'OpenBSD', 'iOS', 'iPadOS', 'Android', 'Unknown'], str
]
Architecture = Annotated[LiteralString, Literal['arm', 'arm64', 'x86', 'x86_64', 'Unknown'], str]
