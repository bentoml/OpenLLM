from __future__ import annotations
import sys
import typing as t

import attr

if t.TYPE_CHECKING:
  from peft.peft_model import PeftModel
  from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast

  from .utils.lazy import VersionInfo
else:
  # NOTE: t.Any is also a type
  PeftModel = PreTrainedModel = PreTrainedTokenizer = PreTrainedTokenizerBase = PreTrainedTokenizerFast = t.Any
  # NOTE: that VersionInfo is from openllm.utils.lazy.VersionInfo
  VersionInfo = t.Any

M = t.TypeVar('M', bound=t.Union[PreTrainedModel, PeftModel])
T = t.TypeVar('T', bound=t.Union[PreTrainedTokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerBase])


def get_literal_args(typ: t.Any) -> tuple[str, ...]:
  return getattr(typ, '__args__', tuple())


AnyCallable = t.Callable[..., t.Any]
DictStrAny = t.Dict[str, t.Any]
ListAny = t.List[t.Any]
ListStr = t.List[str]
TupleAny = t.Tuple[t.Any, ...]
At = t.TypeVar('At', bound=attr.AttrsInstance)

LiteralDtype = t.Literal['float16', 'float32', 'bfloat16', 'int8', 'int16']
LiteralSerialisation = t.Literal['safetensors', 'legacy']
LiteralQuantise = t.Literal['int8', 'int4', 'gptq', 'awq', 'squeezellm']
LiteralBackend = t.Literal['pt', 'vllm', 'ctranslate', 'ggml', 'mlc']
AdapterType = t.Literal[
  'lora', 'adalora', 'adaption_prompt', 'prefix_tuning', 'p_tuning', 'prompt_tuning', 'ia3', 'loha', 'lokr'
]

# TODO: support quay
LiteralContainerRegistry = t.Literal['docker', 'gh', 'ecr']
LiteralContainerVersionStrategy = t.Literal['release', 'nightly', 'latest', 'custom']

LiteralResourceSpec = t.Literal['cloud-tpus.google.com/v2', 'amd.com/gpu', 'nvidia.com/gpu', 'cpu']

InferenceReturnType = t.Literal['text', 'object', 'token']

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
  from typing_extensions import (
    Concatenate as Concatenate,
    ParamSpec as ParamSpec,
    TypeAlias as TypeAlias,
    TypeGuard as TypeGuard,
  )

if sys.version_info[:2] >= (3, 9):
  from typing import Annotated as Annotated
else:
  from typing_extensions import Annotated as Annotated


class AdapterTuple(TupleAny):
  adapter_id: str
  name: str
  config: DictStrAny


AdapterMap = t.Dict[AdapterType, t.Tuple[AdapterTuple, ...]]
