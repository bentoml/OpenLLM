from __future__ import annotations
import sys
import typing as t

import attr


if t.TYPE_CHECKING:
  from peft.peft_model import PeftModel
  from transformers import PreTrainedModel
  from transformers import PreTrainedTokenizer
  from transformers import PreTrainedTokenizerBase
  from transformers import PreTrainedTokenizerFast

  from .utils.lazy import VersionInfo

M = t.TypeVar('M', bound='t.Union[PreTrainedModel, PeftModel]')
T = t.TypeVar('T', bound='t.Union[PreTrainedTokenizerFast, PreTrainedTokenizer, PreTrainedTokenizerBase]')


def get_literal_args(typ: t.Any) -> tuple[str, ...]:
  return getattr(typ, '__args__', tuple())


AnyCallable = t.Callable[..., t.Any]
DictStrAny = t.Dict[str, t.Any]
ListAny = t.List[t.Any]
ListStr = t.List[str]
TupleAny = t.Tuple[t.Any, ...]
At = t.TypeVar('At', bound=attr.AttrsInstance)

LiteralSerialisation = t.Literal['safetensors', 'legacy']
LiteralQuantise = t.Literal['int8', 'int4', 'gptq', 'awq', 'squeezellm']
LiteralBackend = t.Literal['pt', 'vllm', 'ggml', 'mlc']
AdapterType = t.Literal[
  'lora', 'adalora', 'adaption_prompt', 'prefix_tuning', 'p_tuning', 'prompt_tuning', 'ia3', 'loha', 'lokr'
]

# TODO: support quay
LiteralContainerRegistry = t.Literal['docker', 'gh', 'ecr']
LiteralContainerVersionStrategy = t.Literal['release', 'nightly', 'latest', 'custom']

LiteralResourceSpec = t.Literal['cloud-tpus.google.com/v2', 'amd.com/gpu', 'nvidia.com/gpu', 'cpu']

InferenceReturnType = t.Literal['text', 'object', 'token']

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

if sys.version_info[:2] >= (3, 10):
  from typing import Concatenate as Concatenate
  from typing import ParamSpec as ParamSpec
  from typing import TypeAlias as TypeAlias
  from typing import TypeGuard as TypeGuard
else:
  from typing_extensions import Concatenate as Concatenate
  from typing_extensions import ParamSpec as ParamSpec
  from typing_extensions import TypeAlias as TypeAlias
  from typing_extensions import TypeGuard as TypeGuard


class AdapterTuple(TupleAny):
  adapter_id: str
  name: str
  config: DictStrAny


AdapterMap = t.Dict[AdapterType, t.Tuple[AdapterTuple, ...]]


class RefTuple(TupleAny):
  git_hash: str
  version: VersionInfo
  strategy: LiteralContainerVersionStrategy
