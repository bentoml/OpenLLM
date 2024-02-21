from __future__ import annotations
import typing as t
from datetime import datetime

import attr
from cattr import Converter
from cattr.gen import make_dict_structure_fn, make_dict_unstructure_fn

converter = Converter(omit_if_default=True)


def datetime_structure_hook(dt_like: str | datetime | t.Any, _: t.Any) -> datetime:
  if isinstance(dt_like, str):
    return datetime.fromisoformat(dt_like)
  elif isinstance(dt_like, datetime):
    return dt_like
  else:
    raise Exception(f"Unable to parse datetime from '{dt_like}'")


converter.register_structure_hook_factory(
  attr.has,
  lambda cls: make_dict_structure_fn(
    cls, converter, _cattrs_forbid_extra_keys=getattr(cls, '__forbid_extra_keys__', False)
  ),
)
converter.register_unstructure_hook_factory(
  attr.has,
  lambda cls: make_dict_unstructure_fn(
    cls, converter, _cattrs_omit_if_default=getattr(cls, '__omit_if_default__', False)
  ),
)
converter.register_structure_hook(datetime, datetime_structure_hook)
converter.register_unstructure_hook(datetime, lambda dt: dt.isoformat())
