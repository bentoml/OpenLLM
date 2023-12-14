from __future__ import annotations
from typing import Callable, Dict, Tuple, List, Literal, Any, TypeVar
import sys

import attr

M = TypeVar('M')
T = TypeVar('T')


def get_literal_args(typ: Any) -> Tuple[str, ...]:
  return getattr(typ, '__args__', tuple())


AnyCallable = Callable[..., Any]
DictStrAny = Dict[str, Any]
ListStr = List[str]
At = TypeVar('At', bound=attr.AttrsInstance)
LiteralDtype = Literal['float16', 'float32', 'bfloat16', 'int8', 'int16']
LiteralSerialisation = Literal['safetensors', 'legacy']
LiteralQuantise = Literal['int8', 'int4', 'gptq', 'awq', 'squeezellm']
LiteralBackend = Literal['pt', 'vllm', 'ctranslate', 'triton']  # TODO: ggml
AdapterType = Literal['lora', 'adalora', 'adaption_prompt', 'prefix_tuning', 'p_tuning', 'prompt_tuning', 'ia3', 'loha', 'lokr']
LiteralVersionStrategy = Literal['release', 'nightly', 'latest', 'custom']


class AdapterTuple(Tuple[Any, ...]):
  adapter_id: str
  name: str
  config: DictStrAny


AdapterMap = Dict[AdapterType, Tuple[AdapterTuple, ...]]

if sys.version_info[:2] >= (3, 11):
  from typing import (
    LiteralString as LiteralString,
    NotRequired as NotRequired,
    Required as Required,
    Self as Self,
    dataclass_transform as dataclass_transform,
    overload as overload,
  )
else:
  from typing_extensions import (
    LiteralString as LiteralString,
    NotRequired as NotRequired,
    Required as Required,
    Self as Self,
    dataclass_transform as dataclass_transform,
    overload as overload,
  )

if sys.version_info[:2] >= (3, 10):
  from typing import Concatenate as Concatenate, ParamSpec as ParamSpec, TypeAlias as TypeAlias, TypeGuard as TypeGuard
else:
  from typing_extensions import Concatenate as Concatenate, ParamSpec as ParamSpec, TypeAlias as TypeAlias, TypeGuard as TypeGuard

if sys.version_info[:2] >= (3, 9):
  from typing import Annotated as Annotated
else:
  from typing_extensions import Annotated as Annotated
