from __future__ import annotations
import typing as t

from abc import abstractmethod

import attr
import orjson

from openllm_core import utils

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import TypeAlias

ReprArgs: TypeAlias = t.Generator[t.Tuple[t.Optional[str], t.Any], None, None]

class ReprMixin:
  @property
  @abstractmethod
  def __repr_keys__(self) -> set[str]:
    raise NotImplementedError

  '''This can be overriden by base class using this mixin.'''

  def __repr__(self) -> str:
    return f'{self.__class__.__name__} {orjson.dumps({k: utils.bentoml_cattr.unstructure(v) if attr.has(v) else v for k, v in self.__repr_args__()}, option=orjson.OPT_INDENT_2).decode()}'

  '''The `__repr__` for any subclass of Mixin.

  It will print nicely the class name with each of the fields under '__repr_keys__' as kv JSON dict.
  '''

  def __str__(self) -> str:
    return self.__repr_str__(' ')

  '''The string representation of the given Mixin subclass.

  It will contains all of the attributes from __repr_keys__
  '''

  def __repr_name__(self) -> str:
    return self.__class__.__name__

  '''Name of the instance's class, used in __repr__.'''

  def __repr_str__(self, join_str: str) -> str:
    return join_str.join(repr(v) if a is None else f'{a}={v!r}' for a, v in self.__repr_args__())

  '''To be used with __str__.'''

  def __repr_args__(self) -> ReprArgs:
    return ((k, getattr(self, k)) for k in self.__repr_keys__)

  '''This can also be overriden by base class using this mixin.

  By default it does a getattr of the current object from __repr_keys__.
  '''
