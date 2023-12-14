from __future__ import annotations
import typing as t
from abc import abstractmethod
import attr, orjson
from openllm_core import utils

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import TypeAlias
ReprArgs: TypeAlias = t.Generator[t.Tuple[t.Optional[str], t.Any], None, None]


class ReprMixin:
  @property
  @abstractmethod
  def __repr_keys__(self) -> set[str]:
    raise NotImplementedError

  def __repr__(self) -> str:
    return f'{self.__class__.__name__} {orjson.dumps({k: utils.converter.unstructure(v) if attr.has(v) else v for k, v in self.__repr_args__()}, option=orjson.OPT_INDENT_2).decode()}'

  def __str__(self) -> str:
    return self.__repr_str__(' ')

  def __repr_name__(self) -> str:
    return self.__class__.__name__

  def __repr_str__(self, join_str: str) -> str:
    return join_str.join(repr(v) if a is None else f'{a}={v!r}' for a, v in self.__repr_args__())

  def __repr_args__(self) -> ReprArgs:
    return ((k, getattr(self, k)) for k in self.__repr_keys__)
