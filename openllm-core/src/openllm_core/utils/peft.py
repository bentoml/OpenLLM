from __future__ import annotations
import enum
import typing as t

import attr
import inflection
from deepmerge import Merger

from . import dantic
from ..exceptions import ForbiddenAttributeError

config_merger = Merger([(dict, 'merge')], ['override'], ['override'])

if t.TYPE_CHECKING:
  from peft.config import PeftConfig

  from .._configuration import LLMConfig
  from .._typing_compat import AdapterType


# case insensitive, but rename to conform with type
class _PeftEnumMeta(enum.EnumMeta):
  def __getitem__(self, __key: str | t.Any, /) -> t.Any:
    if isinstance(__key, str):
      __key = inflection.underscore(__key).upper()
    return self._member_map_[__key]


# vendorred from peft.utils.config.PeftType since we don't have hard dependency on peft
# see https://github.com/huggingface/peft/blob/main/src/peft/utils/config.py
class PeftType(str, enum.Enum, metaclass=_PeftEnumMeta):
  PROMPT_TUNING = 'PROMPT_TUNING'
  MULTITASK_PROMPT_TUNING = 'MULTITASK_PROMPT_TUNING'
  P_TUNING = 'P_TUNING'
  PREFIX_TUNING = 'PREFIX_TUNING'
  LORA = 'LORA'
  ADALORA = 'ADALORA'
  ADAPTION_PROMPT = 'ADAPTION_PROMPT'
  IA3 = 'IA3'
  LOHA = 'LOHA'
  LOKR = 'LOKR'

  @classmethod
  def _missing_(cls, value: object) -> enum.Enum | None:
    if isinstance(value, str):
      normalized = inflection.underscore(value).upper()
      if normalized in cls._member_map_:
        return cls._member_map_[normalized]
    return None

  @classmethod
  def supported(cls) -> set[str]:
    return {inflection.underscore(v.value) for v in cls}

  @staticmethod
  def get(__key: str | t.Any, /) -> PeftType:
    return PeftType[__key]  # type-safe getitem.


PEFT_TASK_TYPE_TARGET_MAPPING = {'causal_lm': 'CAUSAL_LM', 'seq2seq_lm': 'SEQ_2_SEQ_LM'}

_object_setattr = object.__setattr__


def _adapter_converter(value: AdapterType | str | PeftType | None) -> PeftType:
  if value is None:
    raise ValueError("'AdapterType' cannot be None.")
  if isinstance(value, PeftType):
    return value
  if value not in PeftType.supported():
    raise ValueError(f"Given '{value}' is not a supported adapter type.")
  return PeftType.get(value)


@attr.define(slots=True, init=True)
class FineTuneConfig:
  adapter_type: PeftType = dantic.Field(
    'lora',
    description=f"The type of adapter to use for fine-tuning. Available supported methods: {PeftType.supported()}, default to 'lora'",
    use_default_converter=False,
    converter=_adapter_converter,
  )
  adapter_config: t.Dict[str, t.Any] = dantic.Field(
    None,
    description='The configuration for the adapter. The content of the dict depends on the adapter type.',
    validator=attr.validators.optional(attr.validators.instance_of(dict)),
    converter=attr.converters.default_if_none(factory=dict),
    use_default_converter=False,
  )
  inference_mode: bool = dantic.Field(
    False, description='Whether to use this Adapter for inference', use_default_converter=False
  )
  llm_config_class: type[LLMConfig] = dantic.Field(
    None, description='The reference class to openllm.LLMConfig', use_default_converter=False
  )

  def build(self) -> PeftConfig:
    try:
      from peft.mapping import get_peft_config
      from peft.utils.peft_types import TaskType
    except ImportError:
      raise ImportError('PEFT is not installed. Please install it via `pip install "openllm[fine-tune]"`.') from None
    adapter_config = self.adapter_config.copy()
    # no need for peft_type
    if 'peft_type' in adapter_config:
      adapter_config.pop('peft_type')
    for k in {'enable_lora', 'merge_weights'}:  # these keys are from older PEFT and no longer valid.
      if k in adapter_config:
        adapter_config.pop(k)
    # respect user set task_type if it is passed, otherwise use one managed by OpenLLM
    inference_mode = adapter_config.pop('inference_mode', self.inference_mode)
    task_type = adapter_config.pop('task_type', TaskType[self.llm_config_class.peft_task_type()])
    adapter_config = {
      'peft_type': self.adapter_type.value,
      'task_type': task_type,
      'inference_mode': inference_mode,
      **adapter_config,
    }
    return get_peft_config(adapter_config)

  def train(self) -> FineTuneConfig:
    _object_setattr(self, 'inference_mode', False)
    return self

  def eval(self) -> FineTuneConfig:
    _object_setattr(self, 'inference_mode', True)
    return self

  def with_config(self, **attrs: t.Any) -> FineTuneConfig:
    adapter_type, inference_mode = (
      attrs.pop('adapter_type', self.adapter_type),
      attrs.get('inference_mode', self.inference_mode),
    )
    if 'llm_config_class' in attrs:
      raise ForbiddenAttributeError("'llm_config_class' should not be passed when using 'with_config'.")
    return attr.evolve(
      self,
      adapter_type=adapter_type,
      inference_mode=inference_mode,
      adapter_config=config_merger.merge(self.adapter_config, attrs),
    )

  @classmethod
  def from_config(cls, ft_config: dict[str, t.Any], llm_config_cls: type[LLMConfig]) -> FineTuneConfig:
    copied = ft_config.copy()
    adapter_type = copied.pop('adapter_type', 'lora')
    inference_mode = copied.pop('inference_mode', False)
    llm_config_class = copied.pop('llm_confg_class', llm_config_cls)
    return cls(
      adapter_type=adapter_type,
      adapter_config=copied,
      inference_mode=inference_mode,
      llm_config_class=llm_config_class,
    )
