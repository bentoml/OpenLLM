# mypy: disable-error-code="attr-defined,no-untyped-call,type-var,operator,arg-type,no-redef,misc"
from __future__ import annotations
import copy
import importlib.util
import logging
import os
import sys
import types
import typing as t

import attr
import click_option_group as cog
import inflection
import orjson
from attr._make import _CountingAttr, _make_init, _transform_attrs
from cattr.gen import make_dict_structure_fn, make_dict_unstructure_fn, override
from deepmerge.merger import Merger

import openllm_core

from ._typing_compat import (
  AdapterType,
  AnyCallable,
  At,
  DictStrAny,
  ListStr,
  LiteralBackend,
  LiteralSerialisation,
  LiteralString,
  NotRequired,
  Required,
  Self,
  overload,
)
from .exceptions import ForbiddenAttributeError, MissingDependencyError
from .utils import LazyLoader, ReprMixin, codegen, converter, dantic, field_env_key, first_not_none, lenient_issubclass
from .utils.peft import PEFT_TASK_TYPE_TARGET_MAPPING, FineTuneConfig

if t.TYPE_CHECKING:
  import click
  import transformers
  import vllm
  from attrs import AttrsInstance

  from openllm.protocol.cohere import CohereChatRequest, CohereGenerateRequest
  from openllm.protocol.openai import ChatCompletionRequest, CompletionRequest
else:
  vllm = LazyLoader('vllm', globals(), 'vllm')
  transformers = LazyLoader('transformers', globals(), 'transformers')
  peft = LazyLoader('peft', globals(), 'peft')

__all__ = ['LLMConfig', 'GenerationConfig', 'SamplingParams', 'field_env_key']

logger = logging.getLogger(__name__)
config_merger = Merger([(dict, 'merge')], ['override'], ['override'])
_object_setattr = object.__setattr__


@attr.frozen(slots=True, repr=False, init=False)
class GenerationConfig(ReprMixin):
  """GenerationConfig is the attrs-compatible version of ``transformers.GenerationConfig``, with some additional validation and environment constructor.

  Note that we always set `do_sample=True`. This class is not designed to be used directly, rather
  to be used conjunction with LLMConfig. The instance of the generation config can then be accessed
  via ``LLMConfig.generation_config``.
  """

  max_new_tokens: int = dantic.Field(
    20, ge=0, description='The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.'
  )
  min_length: int = dantic.Field(
    0,
    ge=0,
    description='The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.',
  )
  min_new_tokens: int = dantic.Field(
    description='The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.'
  )
  early_stopping: bool = dantic.Field(
    False,
    description="""Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values: `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; `"never"`, where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm) """,
  )
  max_time: float = dantic.Field(
    description='The maximum amount of time you allow the computation to run for in seconds. generation will still finish the current pass after allocated time has been passed.'
  )
  num_beams: int = dantic.Field(1, description='Number of beams for beam search. 1 means no beam search.')
  num_beam_groups: int = dantic.Field(
    1,
    description='Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.',
  )
  penalty_alpha: float = dantic.Field(
    description='The values balance the model confidence and the degeneration penalty in contrastive search decoding.'
  )
  use_cache: bool = dantic.Field(
    True,
    description='Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.',
  )
  temperature: float = dantic.Field(
    1.0, ge=0.0, le=1.0, description='The value used to modulate the next token probabilities.'
  )
  top_k: int = dantic.Field(
    50, description='The number of highest probability vocabulary tokens to keep for top-k-filtering.'
  )
  top_p: float = dantic.Field(
    1.0,
    description='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.',
  )
  typical_p: float = dantic.Field(
    1.0,
    description='Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to `typical_p` or higher are kept for generation. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.',
  )
  epsilon_cutoff: float = dantic.Field(
    0.0,
    description='If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more details.',
  )
  eta_cutoff: float = dantic.Field(
    0.0,
    description='Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between 0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more details. ',
  )
  diversity_penalty: float = dantic.Field(
    0.0,
    description="This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled. ",
  )
  repetition_penalty: float = dantic.Field(
    1.0,
    description='The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.',
  )
  encoder_repetition_penalty: float = dantic.Field(
    1.0,
    description='The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the original input. 1.0 means no penalty.',
  )
  length_penalty: float = dantic.Field(
    1.0,
    description='Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.',
  )
  no_repeat_ngram_size: int = dantic.Field(
    0, description='If set to int > 0, all ngrams of that size can only occur once.'
  )
  bad_words_ids: t.List[t.List[int]] = dantic.Field(
    description='List of token ids that are not allowed to be generated. In order to get the token ids of the words that should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids`.'
  )
  force_words_ids: t.Union[t.List[t.List[int]], t.List[t.List[t.List[int]]]] = dantic.Field(
    description='List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple list of words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`, this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one can allow different forms of each word. '
  )
  renormalize_logits: bool = dantic.Field(
    False,
    description="Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization. ",
  )
  forced_bos_token_id: int = dantic.Field(
    description='The id of the token to force as the first generated token after the ``decoder_start_token_id``. Useful for multilingual models like [mBART](https://huggingface.co/docs/transformers/model_doc/mbart) where the first generated token needs to be the target language token. '
  )
  forced_eos_token_id: t.Union[int, t.List[int]] = dantic.Field(
    description='The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a list to set multiple *end-of-sequence* tokens.'
  )
  remove_invalid_values: bool = dantic.Field(
    False,
    description='Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash. Note that using `remove_invalid_values` can slow down generation.',
  )
  exponential_decay_length_penalty: t.Tuple[int, float] = dantic.Field(
    description='This tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay'
  )
  suppress_tokens: t.List[int] = dantic.Field(
    description='A list of tokens that will be suppressed at generation. The `SupressTokens` logit processor will set their log probs to `-inf` so that they are not sampled.'
  )
  begin_suppress_tokens: t.List[int] = dantic.Field(
    description='A list of tokens that will be suppressed at the beginning of the generation. The `SupressBeginTokens` logit processor will set their log probs to `-inf` so that they are not sampled. '
  )
  forced_decoder_ids: t.List[t.List[int]] = dantic.Field(
    description='A list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling. For example, `[[1, 123]]` means the second generated token will always be a token of index 123.'
  )
  num_return_sequences: int = dantic.Field(
    1, description='The number of independently computed returned sequences for each element in the batch.'
  )
  output_attentions: bool = dantic.Field(
    False,
    description='Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more details.',
  )
  output_hidden_states: bool = dantic.Field(
    False,
    description='Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more details.',
  )
  output_scores: bool = dantic.Field(
    False,
    description='Whether or not to return the prediction scores. See `scores` under returned tensors for more details.',
  )
  pad_token_id: int = dantic.Field(description='The id of the *padding* token.')
  bos_token_id: int = dantic.Field(description='The id of the *beginning-of-sequence* token.')
  eos_token_id: t.Union[int, t.List[int]] = dantic.Field(
    description='The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.'
  )
  encoder_no_repeat_ngram_size: int = dantic.Field(
    0,
    description='If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`.',
  )
  decoder_start_token_id: int = dantic.Field(
    description='If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.'
  )

  def __init__(self, *, _internal: bool = False, **attrs: t.Any):
    if not _internal:
      raise RuntimeError(
        'GenerationConfig is not meant to be used directly, but you can access this via a LLMConfig.generation_config'
      )
    self.__attrs_init__(**attrs)

  def __getitem__(self, item: str) -> t.Any:
    if hasattr(self, item):
      return getattr(self, item)
    raise KeyError(f"'{self.__class__.__name__}' has no attribute {item}.")

  @property
  def __repr_keys__(self) -> set[str]:
    return {i.name for i in attr.fields(self.__class__)}


converter.register_unstructure_hook_factory(
  lambda cls: attr.has(cls) and lenient_issubclass(cls, GenerationConfig),
  lambda cls: make_dict_unstructure_fn(
    cls,
    converter,
    _cattrs_omit_if_default=False,
    _cattrs_use_linecache=True,
    **{k: override(omit=True) for k, v in attr.fields_dict(cls).items() if v.default in (None, attr.NOTHING)},
  ),
)

_GenerationConfigT = t.TypeVar('_GenerationConfig', bound=GenerationConfig)


@attr.frozen(slots=True, repr=False, init=False)
class SamplingParams(ReprMixin):
  """SamplingParams is the attr-compatible version of ``vllm.SamplingParams``. It provides some utilities to also respect shared variables from ``openllm.LLMConfig``.

  The following value will be parsed directly from ``openllm.LLMConfig``:
  - temperature
  - top_k
  - top_p
  - max_tokens -> max_new_tokens
  """

  n: int = dantic.Field(1, description='Number of output sequences to return for the given prompt.')
  best_of: int = dantic.Field(
    None,
    description='Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`.',
  )
  presence_penalty: float = dantic.Field(
    0.0,
    description='Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.',
  )
  frequency_penalty: float = dantic.Field(
    0.0,
    description='Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.',
  )
  use_beam_search: bool = dantic.Field(False, description='Whether to use beam search instead of sampling.')
  ignore_eos: bool = dantic.Field(
    False,
    description='Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.',
  )
  logprobs: int = dantic.Field(None, description='Number of log probabilities to return per output token.')
  prompt_logprobs: t.Optional[int] = dantic.Field(
    None, description='Number of log probabilities to return per input token.'
  )
  skip_special_tokens: bool = dantic.Field(True, description='Whether to skip special tokens in the generated output.')
  # space_between_special_tokens: bool = dantic.Field(True, description='Whether to add a space between special tokens in the generated output.')

  if t.TYPE_CHECKING:
    max_tokens: int
    temperature: float
    top_k: int
    top_p: float

  def __init__(self, *, _internal: bool = False, **attrs: t.Any):
    if not _internal:
      raise RuntimeError(
        'SamplingParams is not meant to be used directly, but you can access this via a LLMConfig.sampling_config.'
      )
    _object_setattr(self, 'max_tokens', attrs.pop('max_tokens', 16))
    _object_setattr(self, 'temperature', attrs.pop('temperature', 1.0))
    _object_setattr(self, 'top_k', attrs.pop('top_k', -1))
    _object_setattr(self, 'top_p', attrs.pop('top_p', 1.0))
    _object_setattr(self, 'repetition_penalty', attrs.pop('repetition_penalty', 1.0))
    _object_setattr(self, 'length_penalty', attrs.pop('length_penalty', 1.0))
    _object_setattr(self, 'early_stopping', attrs.pop('early_stopping', False))
    self.__attrs_init__(**attrs)

  def __getitem__(self, item: str) -> t.Any:
    if hasattr(self, item):
      return getattr(self, item)
    raise KeyError(f"'{self.__class__.__name__}' has no attribute {item}.")

  @property
  def __repr_keys__(self) -> set[str]:
    return {i.name for i in attr.fields(self.__class__)}

  def build(self) -> vllm.SamplingParams:
    return vllm.SamplingParams(
      max_tokens=self.max_tokens,
      early_stopping=self.early_stopping,
      length_penalty=self.length_penalty,
      temperature=self.temperature,
      top_k=self.top_k,
      top_p=self.top_p,
      **converter.unstructure(self),
    )

  @classmethod
  def from_generation_config(cls, generation_config: GenerationConfig, **attrs: t.Any) -> Self:
    """The main entrypoint for creating a SamplingParams from ``openllm.LLMConfig``."""
    if 'max_tokens' in attrs and 'max_new_tokens' in attrs:
      raise ValueError("Both 'max_tokens' and 'max_new_tokens' are passed. Make sure to only use one of them.")
    temperature = first_not_none(attrs.pop('temperature', None), default=generation_config['temperature'])
    top_k = first_not_none(attrs.pop('top_k', None), default=generation_config['top_k'])
    top_p = first_not_none(attrs.pop('top_p', None), default=generation_config['top_p'])
    max_tokens = first_not_none(
      attrs.pop('max_tokens', None), attrs.pop('max_new_tokens', None), default=generation_config['max_new_tokens']
    )
    repetition_penalty = first_not_none(
      attrs.pop('repetition_penalty', None), default=generation_config['repetition_penalty']
    )
    length_penalty = first_not_none(attrs.pop('length_penalty', None), default=generation_config['length_penalty'])
    early_stopping = first_not_none(attrs.pop('early_stopping', None), default=generation_config['early_stopping'])
    logprobs = first_not_none(attrs.pop('logprobs', None), default=None)
    prompt_logprobs = first_not_none(attrs.pop('prompt_logprobs', None), default=None)

    return cls(
      _internal=True,
      temperature=temperature,
      top_k=top_k,
      top_p=top_p,
      max_tokens=max_tokens,
      repetition_penalty=repetition_penalty,
      length_penalty=length_penalty,
      early_stopping=early_stopping,
      logprobs=logprobs,
      prompt_logprobs=prompt_logprobs,
      **attrs,
    )


converter.register_unstructure_hook_factory(
  lambda cls: attr.has(cls) and lenient_issubclass(cls, SamplingParams),
  lambda cls: make_dict_unstructure_fn(
    cls,
    converter,
    _cattrs_omit_if_default=False,
    _cattrs_use_linecache=True,
    **{
      k: override(omit_if_default=True) for k, v in attr.fields_dict(cls).items() if v.default in (None, attr.NOTHING)
    },
  ),
)
converter.register_structure_hook_factory(
  lambda cls: attr.has(cls) and lenient_issubclass(cls, SamplingParams),
  lambda cls: make_dict_structure_fn(
    cls, converter, _cattrs_forbid_extra_keys=True, max_new_tokens=override(rename='max_tokens')
  ),
)

_SamplingParamsT = t.TypeVar('_SamplingParams', bound=SamplingParams)

# cached it here to save one lookup per assignment
_object_getattribute = object.__getattribute__


class ModelSettings(t.TypedDict, total=False):
  """ModelSettings serve only for typing purposes as this is transcribed into LLMConfig.__config__.

  Note that all fields from this dictionary will then be converted to __openllm_*__ fields in LLMConfig.

  If the field below changes, make sure to run ./tools/update-config-stubs.py to generate correct __getitem__
  stubs for type-checking purposes.
  """

  # NOTE: These required fields should be at the top, as it will be kw_only
  default_id: Required[str]
  model_ids: Required[ListStr]
  architecture: Required[str]

  # meta
  url: str
  serialisation: LiteralSerialisation
  trust_remote_code: bool
  service_name: NotRequired[str]
  requirements: t.Optional[ListStr]

  # llm implementation specifics
  model_type: t.Literal['causal_lm', 'seq2seq_lm']

  # naming convention, only name_type is needed to infer from the class
  # as the three below it can be determined automatically
  name_type: NotRequired[t.Optional[t.Literal['dasherize', 'lowercase']]]
  backend: t.Tuple[LiteralBackend, ...]
  model_name: NotRequired[str]
  start_name: NotRequired[str]
  # serving configuration
  timeout: int
  workers_per_resource: t.Union[int, float]

  # the target generation_config class to be used.
  fine_tune_strategies: t.Tuple[t.Dict[str, t.Any], ...]

  # Chat models related configuration
  add_generation_prompt: bool


_transformed_type: DictStrAny = {'fine_tune_strategies': t.Dict[AdapterType, FineTuneConfig]}


@attr.define(
  frozen=False,
  slots=True,
  field_transformer=lambda _, __: [
    attr.Attribute.from_counting_attr(
      k,
      dantic.Field(
        kw_only=False if t.get_origin(ann) is not Required else True,
        auto_default=True,
        use_default_converter=False,
        type=_transformed_type.get(k, ann),
        metadata={'target': f'__openllm_{k}__'},
        description=f'ModelSettings field for {k}.',
      ),
    )
    for k, ann in t.get_type_hints(ModelSettings).items()
  ],
)
class _ModelSettingsAttr:
  def __getitem__(self, key: str) -> t.Any:
    if key in codegen.get_annotations(ModelSettings):
      return _object_getattribute(self, key)
    raise KeyError(key)

  # NOTE: The below are dynamically generated by the field_transformer
  if t.TYPE_CHECKING:
    # fmt: off
    # update-config-stubs.py: attrs start
    default_id:str
    model_ids:ListStr
    architecture:str
    url:str
    serialisation:LiteralSerialisation
    trust_remote_code:bool
    service_name:str
    requirements:t.Optional[ListStr]
    model_type:t.Literal['causal_lm', 'seq2seq_lm']
    name_type:t.Optional[t.Literal['dasherize', 'lowercase']]
    backend:t.Tuple[LiteralBackend, ...]
    model_name:str
    start_name:str
    timeout:int
    workers_per_resource:t.Union[int, float]
    fine_tune_strategies:t.Dict[AdapterType, FineTuneConfig]
    add_generation_prompt:bool
    # update-config-stubs.py: attrs stop
    # fmt: on


_DEFAULT = _ModelSettingsAttr(
  **ModelSettings(
    default_id='__default__',
    model_ids=['__default__'],
    architecture='PreTrainedModel',
    serialisation='legacy',
    backend=('pt', 'vllm'),
    name_type='dasherize',
    url='',
    model_type='causal_lm',
    trust_remote_code=False,
    requirements=None,
    add_generation_prompt=False,
    timeout=int(36e6),
    service_name='',
    workers_per_resource=1.0,
  )
)


def structure_settings(cls: type[LLMConfig], _: type[_ModelSettingsAttr]) -> _ModelSettingsAttr:
  _cl_name = cls.__name__.replace('Config', '')
  has_custom_name = all(i in cls.__config__ for i in {'model_name', 'start_name'})
  _config = attr.evolve(_DEFAULT, **cls.__config__)
  _attr = {}

  if not has_custom_name:
    _attr['model_name'] = inflection.underscore(_cl_name) if _config['name_type'] == 'dasherize' else _cl_name.lower()
    _attr['start_name'] = (
      inflection.dasherize(_attr['model_name']) if _config['name_type'] == 'dasherize' else _attr['model_name']
    )
  model_name = _attr['model_name'] if 'model_name' in _attr else _config.model_name

  _attr.update(
    {
      'service_name': f'generated_{model_name}_service.py',
      'fine_tune_strategies': {
        ft_config.get('adapter_type', 'lora'): FineTuneConfig.from_config(ft_config, cls)
        for ft_config in _config.fine_tune_strategies  # ft_config is a dict here before transformer
      }
      if _config.fine_tune_strategies
      else {},
    }
  )

  return attr.evolve(_config, **_attr)


converter.register_structure_hook(_ModelSettingsAttr, structure_settings)


def _setattr_class(attr_name: str, value_var: t.Any) -> str:
  return f"setattr(cls, '{attr_name}', {value_var})"


def _make_assignment_script(
  cls: type[LLMConfig], attributes: attr.AttrsInstance, _prefix: LiteralString = 'openllm'
) -> t.Callable[..., None]:
  """Generate the assignment script with prefix attributes __openllm_<value>__."""
  args: ListStr = []
  globs: DictStrAny = {
    'cls': cls,
    '_cached_attribute': attributes,
    '_cached_getattribute_get': _object_getattribute.__get__,
  }
  annotations: DictStrAny = {'return': None}

  lines: ListStr = []
  for attr_name, field in attr.fields_dict(attributes.__class__).items():
    arg_name = field.metadata.get('target', f'__{_prefix}_{inflection.underscore(attr_name)}__')
    args.append(f"{attr_name}=getattr(_cached_attribute, '{attr_name}')")
    lines.append(_setattr_class(arg_name, attr_name))
    annotations[attr_name] = field.type

  return codegen.generate_function(
    cls, '__assign_attr', lines, args=('cls', *args), globs=globs, annotations=annotations
  )


_reserved_namespace = {'__config__', 'GenerationConfig', 'SamplingParams'}


@attr.define(slots=True)
class _ConfigAttr(t.Generic[_GenerationConfigT, _SamplingParamsT]):
  @staticmethod
  def Field(default: t.Any = None, **attrs: t.Any) -> t.Any:
    """Field is a alias to the internal dantic utilities to easily create
    attrs.fields with pydantic-compatible interface. For example:

    ```python
    class MyModelConfig(openllm.LLMConfig):
        field1 = openllm.LLMConfig.Field(...)
    ```
    """
    return dantic.Field(default, **attrs)

  # NOTE: The following is handled via __init_subclass__, and is only used for TYPE_CHECKING
  if t.TYPE_CHECKING:
    # NOTE: public attributes to override
    __config__: ModelSettings = Field(None)
    """Internal configuration for this LLM model. Each of the field in here will be populated
        and prefixed with __openllm_<value>__"""
    GenerationConfig: _GenerationConfigT = Field(None)
    """Users can override this subclass of any given LLMConfig to provide GenerationConfig
        default value. For example:

        ```python
        class MyAwesomeModelConfig(openllm.LLMConfig):
            class GenerationConfig:
                max_new_tokens: int = 200
                top_k: int = 10
                num_return_sequences: int = 1
                eos_token_id: int = 11
        ```
        """
    SamplingParams: _SamplingParamsT = Field(None)
    """Users can override this subclass of any given LLMConfig to provide SamplingParams
        default value. For example:

        ```python
        class MyAwesomeModelConfig(openllm.LLMConfig):
            class SamplingParams:
                max_new_tokens: int = 200
                top_k: int = 10
                num_return_sequences: int = 1
                eos_token_id: int = 11
        ```
        """
    # NOTE: Internal attributes that should only be used by OpenLLM. Users usually shouldn't
    # concern any of these. These are here for pyright not to complain.
    __attrs_attrs__: tuple[attr.Attribute[t.Any], ...] = Field(None, init=False)
    """Since we are writing our own __init_subclass__, which is an alternative way for __prepare__,
        we want openllm.LLMConfig to be attrs-like dataclass that has pydantic-like interface.
        __attrs_attrs__ will be handled dynamically by __init_subclass__.
        """
    __openllm_hints__: DictStrAny = Field(None, init=False)
    """An internal cache of resolved types for this LLMConfig."""
    __openllm_accepted_keys__: set[str] = Field(None, init=False)
    """The accepted keys for this LLMConfig."""
    __openllm_extras__: DictStrAny = Field(None, init=False)
    """Extra metadata for this LLMConfig."""
    __openllm_config_override__: DictStrAny = Field(None, init=False)
    """Additional override for some variables in LLMConfig.__config__"""
    __openllm_generation_class__: type[_GenerationConfigT] = Field(None)
    """The result generated GenerationConfig class for this LLMConfig. This will be used
        to create the generation_config argument that can be used throughout the lifecycle.
        This class will also be managed internally by OpenLLM."""
    __openllm_sampling_class__: type[_SamplingParamsT] = Field(None)
    """The result generated SamplingParams class for this LLMConfig. This will be used
        to create arguments for vLLM LLMEngine that can be used throughout the lifecycle.
        This class will also be managed internally by OpenLLM."""

    def __attrs_init__(self, *args: t.Any, **attrs: t.Any) -> None:
      """Generated __attrs_init__ for LLMConfig subclass that follows the attrs contract."""

    # NOTE: The following will be populated from __config__ and also
    # considered to be public API. Users can also access these via self[key]
    # To update the docstring for these field, update it through tools/update-config-stubs.py

    # fmt: off
    # update-config-stubs.py: special start
    __openllm_default_id__:str=Field(None)
    '''Return the default model to use when using 'openllm start <model_id>'.
        This could be one of the keys in 'self.model_ids' or custom users model.

        This field is required when defining under '__config__'.
        '''
    __openllm_model_ids__:ListStr=Field(None)
    '''A list of supported pretrained models tag for this given runnable.

        For example:
            For FLAN-T5 impl, this would be ["google/flan-t5-small", "google/flan-t5-base",
                                            "google/flan-t5-large", "google/flan-t5-xl", "google/flan-t5-xxl"]

        This field is required when defining under '__config__'.
        '''
    __openllm_architecture__:str=Field(None)
    '''The model architecture that is supported by this LLM.

        Note that any model weights within this architecture generation can always be run and supported by this LLM.

        For example:
            For GPT-NeoX implementation, it is based on GptNeoXForCausalLM, which supports dolly-v2, stablelm:

            ```bash
            openllm start gpt-neox --model-id stabilityai/stablelm-tuned-alpha-3b
            ```'''
    __openllm_url__:str=Field(None)
    '''The resolved url for this LLMConfig.'''
    __openllm_serialisation__:LiteralSerialisation=Field(None)
    '''Default serialisation format for different models. Some will default to use the legacy 'bin'. '''
    __openllm_trust_remote_code__:bool=Field(None)
    '''Whether to always trust remote code'''
    __openllm_service_name__:str=Field(None)
    '''Generated service name for this LLMConfig. By default, it is "generated_{model_name}_service.py"'''
    __openllm_requirements__:t.Optional[ListStr]=Field(None)
    '''The default PyPI requirements needed to run this given LLM. By default, we will depend on bentoml, torch, transformers.'''
    __openllm_model_type__:t.Literal['causal_lm', 'seq2seq_lm']=Field(None)
    '''The model type for this given LLM. By default, it should be causal language modeling. Currently supported "causal_lm" or "seq2seq_lm"'''
    __openllm_name_type__:t.Optional[t.Literal['dasherize', 'lowercase']]=Field(None)
    '''The default name typed for this model. "dasherize" will convert the name to lowercase and
        replace spaces with dashes. "lowercase" will convert the name to lowercase. If this is not set, then both
        `model_name` and `start_name` must be specified.'''
    __openllm_backend__:t.Tuple[LiteralBackend, ...]=Field(None)
    '''List of supported backend for this given LLM class. Currently, we support "pt" and "vllm".'''
    __openllm_model_name__:str=Field(None)
    '''The normalized version of __openllm_start_name__, determined by __openllm_name_type__'''
    __openllm_start_name__:str=Field(None)
    '''Default name to be used with `openllm start`'''
    __openllm_timeout__:int=Field(None)
    '''The default timeout to be set for this given LLM.'''
    __openllm_workers_per_resource__:t.Union[int, float]=Field(None)
    '''The number of workers per resource. This is used to determine the number of workers to use for this model.
        For example, if this is set to 0.5, then OpenLLM will use 1 worker per 2 resources. If this is set to 1, then
        OpenLLM will use 1 worker per resource. If this is set to 2, then OpenLLM will use 2 workers per resource.

        See StarCoder for more advanced usage. See
        https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy for more details.

        By default, it is set to 1.
        '''
    __openllm_fine_tune_strategies__:t.Dict[AdapterType, FineTuneConfig]=Field(None)
    '''The fine-tune strategies for this given LLM.'''
    __openllm_add_generation_prompt__:bool=Field(None)
    '''Whether to add generation prompt token for formatting chat templates. This arguments will be used for chat-based models.'''
    # update-config-stubs.py: special stop
    # fmt: on


class _ConfigBuilder:
  __slots__ = (
    '_cls',
    '_cls_dict',
    '_attr_names',
    '_attrs',
    '_model_name',
    '_base_attr_map',
    '_base_names',
    '_has_pre_init',
    '_has_post_init',
  )

  def __init__(
    self,
    cls: type[LLMConfig],
    these: dict[str, _CountingAttr],
    auto_attribs: bool = False,
    kw_only: bool = False,
    collect_by_mro: bool = True,
  ):
    attrs, base_attrs, base_attr_map = _transform_attrs(
      cls,
      these,
      auto_attribs,
      kw_only,
      collect_by_mro,
      field_transformer=codegen.make_env_transformer(cls, cls.__openllm_model_name__),
    )
    self._cls, self._model_name, self._cls_dict = cls, cls.__openllm_model_name__, dict(cls.__dict__)
    self._attrs = attrs
    self._base_attr_map = base_attr_map
    self._base_names = {a.name for a in base_attrs}
    self._attr_names = tuple(a.name for a in attrs)
    self._has_pre_init = bool(getattr(cls, '__attrs_pre_init__', False))
    self._has_post_init = bool(getattr(cls, '__attrs_post_init__', False))
    self._cls_dict['__attrs_attrs__'] = self._attrs

  def build_class(self) -> type[LLMConfig]:
    """Finalize class based on the accumulated configuration.

    Builder cannot be used after calling this method.

    > A difference between this and attrs._ClassBuilder is that we don't
    > create a new class after constructing all __dict__. This has to do
    > with recursive called within __init_subclass__
    """
    filtered = (*self._attr_names, '__dict__', '__weakref__')
    cd = {k: v for k, v in self._cls_dict.items() if k not in filtered}
    # Traverse the MRO to collect existing slots
    # and check for an existing __weakref__.
    weakref_inherited = False
    existing_slots: DictStrAny = {}
    for base_cls in self._cls.__mro__[1:-1]:
      if base_cls.__dict__.get('__weakref__', None) is not None:
        weakref_inherited = True
      existing_slots.update(
        {name: getattr(base_cls, name, codegen._sentinel) for name in getattr(base_cls, '__slots__', [])}
      )

    names = self._attr_names
    base_names = set(self._base_names)
    if (
      '__weakref__' not in getattr(self._cls, '__slots__', ()) and '__weakref__' not in names and not weakref_inherited
    ):
      names += ('__weakref__',)
    # We only add the names of attributes that aren't inherited.
    # Setting __slots__ to inherited attributes wastes memory.
    slot_names = [name for name in names if name not in base_names]
    # There are slots for attributes from current class
    # that are defined in parent classes.
    # As their descriptors may be overridden by a child class,
    # we collect them here and update the class dict
    reused_slots = {slot: slot_descriptor for slot, slot_descriptor in existing_slots.items() if slot in slot_names}
    # We only add the names of attributes that aren't inherited.
    # Setting __slots__ to inherited attributes wastes memory.
    # __openllm_extras__ holds additional metadata that might be usefule for users, hence we add it to slots
    slot_names = [name for name in slot_names if name not in reused_slots]
    cd.update(reused_slots)
    cd['__slots__'] = tuple(slot_names)
    cd['__qualname__'] = self._cls.__qualname__

    # We can only patch the class here, rather than instantiate
    # a new one, since type.__new__ actually will invoke __init_subclass__
    # and since we use the _ConfigBuilder in __init_subclass__, it will
    # raise recusion error. See https://peps.python.org/pep-0487/ for more
    # information on how __init_subclass__ works.
    for k, value in cd.items():
      setattr(self._cls, k, value)
    return self.make_closure(self._cls)

  def make_closure(self, cls: type[t.Any]) -> type[t.Any]:
    # The following is a fix for
    # <https://github.com/python-attrs/attrs/issues/102>.
    # If a method mentions `__class__` or uses the no-arg super(), the
    # compiler will bake a reference to the class in the method itself
    # as `method.__closure__`.  Since we replace the class with a
    # clone, we rewrite these references so it keeps working.
    for item in cls.__dict__.values():
      if isinstance(item, (classmethod, staticmethod)):
        # Class- and staticmethods hide their functions inside.
        # These might need to be rewritten as well.
        closure_cells = getattr(item.__func__, '__closure__', None)
      elif isinstance(item, property):
        # Workaround for property `super()` shortcut (PY3-only).
        # There is no universal way for other descriptors.
        closure_cells = getattr(item.fget, '__closure__', None)
      else:
        closure_cells = getattr(item, '__closure__', None)

      if not closure_cells:
        continue  # Catch None or the empty list.
      for cell in closure_cells:
        try:
          match = cell.cell_contents is self._cls
        except ValueError:  # noqa: PERF203
          pass  # ValueError: Cell is empty
        else:
          if match:
            cell.cell_contents = cls
    return cls

  def add_attrs_init(self) -> Self:
    self._cls_dict['__attrs_init__'] = codegen.add_method_dunders(
      self._cls,
      _make_init(
        self._cls,
        self._attrs,
        self._has_pre_init,
        self._has_post_init,
        False,
        True,
        False,
        self._base_attr_map,
        False,
        None,
        True,
      ),
    )
    return self

  def add_repr(self) -> Self:
    for key, fn in ReprMixin.__dict__.items():
      if key in ('__repr__', '__str__', '__repr_name__', '__repr_str__', '__repr_args__'):
        self._cls_dict[key] = codegen.add_method_dunders(self._cls, fn)
    self._cls_dict['__repr_keys__'] = property(
      lambda _: {i.name for i in self._attrs} | {'generation_config', 'sampling_config'}
    )
    return self


@attr.define(slots=True, init=False)
class LLMConfig(_ConfigAttr[t.Any, t.Any]):
  """``openllm.LLMConfig`` is a pydantic-like ``attrs`` interface that offers fast and easy-to-use APIs.

  It lives in between the nice UX of `pydantic` and fast performance of `attrs` where it allows users to quickly formulate
  a LLMConfig for any LLM without worrying too much about performance. It does a few things:

  - Automatic environment conversion: Each fields will automatically be provisioned with an environment
  variable, make it easy to work with ahead-of-time or during serving time
  - Familiar API: It is compatible with cattrs as well as providing a few Pydantic-2 like API,
  i.e: ``model_construct_env``, ``to_generation_config``, ``to_click_options``
  - Automatic CLI generation: It can identify each fields and convert it to compatible Click options.
  This means developers can use any of the LLMConfig to create CLI with compatible-Python
  CLI library (click, typer, ...)

  > Internally, LLMConfig is an attrs class. All subclass of LLMConfig contains "attrs-like" features,
  > which means LLMConfig will actually generate subclass to have attrs-compatible API, so that the subclass
  > can be written as any normal Python class.

  To directly configure GenerationConfig for any given LLM, create a GenerationConfig under the subclass:

  ```python
  class FlanT5Config(openllm.LLMConfig):
    class GenerationConfig:
      temperature: float = 0.75
      max_new_tokens: int = 3000
      top_k: int = 50
      top_p: float = 0.4
      repetition_penalty = 1.0
  ```
  By doing so, openllm.LLMConfig will create a compatible GenerationConfig attrs class that can be converted
  to ``transformers.GenerationConfig``. These attribute can be accessed via ``LLMConfig.generation_config``.

  By default, all LLMConfig must provide a __config__ with 'default_id' and 'model_ids'.

  All other fields are optional, and will be use default value if not set.

  ```python
  class FalconConfig(openllm.LLMConfig):
      __config__ = {
          "name_type": "lowercase",
          "trust_remote_code": True,
          "timeout": 3600000,
          "url": "https://falconllm.tii.ae/",
          "requirements": ["einops", "xformers", "safetensors"],
          # NOTE: The below are always required
          "default_id": "tiiuae/falcon-7b",
          "model_ids": [
              "tiiuae/falcon-7b",
              "tiiuae/falcon-40b",
              "tiiuae/falcon-7b-instruct",
              "tiiuae/falcon-40b-instruct",
          ],
      }
  ```

  > **Changelog**:
  > Since 0.1.7, one can also define given fine-tune strategies for given LLM via its config:
  ```python
  class OPTConfig(openllm.LLMConfig):
      __config__ = {
          "name_type": "lowercase",
          "trust_remote_code": False,
          "url": "https://huggingface.co/docs/transformers/model_doc/opt",
          "default_id": "facebook/opt-1.3b",
          "model_ids": [
              "facebook/opt-125m",
              "facebook/opt-350m",
              "facebook/opt-1.3b",
              "facebook/opt-2.7b",
              "facebook/opt-6.7b",
              "facebook/opt-66b",
          ],
          "fine_tune_strategies": (
              {
                  "adapter_type": "lora",
                  "r": 16,
                  "lora_alpha": 32,
                  "target_modules": ["q_proj", "v_proj"],
                  "lora_dropout": 0.05,
                  "bias": "none",
              },
          ),
      }
  ```

  Future work:
  - Support pydantic-core as validation backend.
  """

  def __init_subclass__(cls, **_: t.Any):
    """The purpose of this ``__init_subclass__`` is to offer pydantic UX while adhering to attrs contract.

    This means we will construct all fields and metadata and hack into
    how attrs use some of the 'magic' construction to generate the fields.

    It also does a few more extra features: It also generate all __openllm_*__ config from
    ModelSettings (derived from __config__) to the class.
    """
    if not cls.__name__.endswith('Config'):
      logger.warning("LLMConfig subclass should end with 'Config'. Updating to %sConfig", cls.__name__)
      cls.__name__ = f'{cls.__name__}Config'

    if not hasattr(cls, '__config__'):
      raise RuntimeError("Given LLMConfig must have '__config__' that is not None defined.")

    # auto assignment attributes generated from __config__ after create the new slot class.
    _make_assignment_script(cls, converter.structure(cls, _ModelSettingsAttr))(cls)

    def _make_subclass(
      class_attr: str, base: type[At], globs: dict[str, t.Any] | None = None, suffix_env: LiteralString | None = None
    ) -> type[At]:
      camel_name = cls.__name__.replace('Config', '')
      klass = attr.make_class(
        f'{camel_name}{class_attr}',
        [],
        bases=(base,),
        slots=True,
        weakref_slot=True,
        frozen=True,
        repr=False,
        init=False,
        collect_by_mro=True,
        field_transformer=codegen.make_env_transformer(
          cls,
          cls.__openllm_model_name__,
          suffix=suffix_env,
          globs=globs,
          default_callback=lambda field_name, field_default: getattr(
            getattr(cls, class_attr), field_name, field_default
          )
          if codegen.has_own_attribute(cls, class_attr)
          else field_default,
        ),
      )
      # For pickling to work, the __module__ variable needs to be set to the
      # frame where the class is created. This respect the module that is created from cls
      try:
        klass.__module__ = cls.__module__
      except (AttributeError, ValueError):
        pass
      return t.cast('type[At]', klass)

    cls.__openllm_generation_class__ = _make_subclass('GenerationConfig', GenerationConfig, suffix_env='generation')
    cls.__openllm_sampling_class__ = _make_subclass('SamplingParams', SamplingParams, suffix_env='sampling')

    # process a fields under cls.__dict__ and auto convert them with dantic.Field
    # this is similar logic to attr._make._transform_attrs
    cd = cls.__dict__
    anns = codegen.get_annotations(cls)
    # _CountingAttr is the underlying representation of attr.field
    ca_names = {name for name, attr in cd.items() if isinstance(attr, _CountingAttr)}
    these: dict[str, _CountingAttr] = {}
    annotated_names: set[str] = set()
    for attr_name, typ in anns.items():
      if codegen.is_class_var(typ):
        continue
      annotated_names.add(attr_name)
      val = cd.get(attr_name, attr.NOTHING)
      if not isinstance(val, _CountingAttr):
        if val is attr.NOTHING:
          val = cls.Field(env=field_env_key(attr_name))
        else:
          val = cls.Field(default=val, env=field_env_key(attr_name))
      these[attr_name] = val
    unannotated = ca_names - annotated_names
    if len(unannotated) > 0:
      missing_annotated = sorted(unannotated, key=lambda n: t.cast('_CountingAttr', cd.get(n)).counter)
      raise openllm_core.exceptions.MissingAnnotationAttributeError(
        f"The following field doesn't have a type annotation: {missing_annotated}"
      )
    # We need to set the accepted key before generation_config
    # as generation_config is a special field that users shouldn't pass.
    cls.__openllm_accepted_keys__ = (
      set(these.keys())
      | {a.name for a in attr.fields(cls.__openllm_generation_class__)}
      | {a.name for a in attr.fields(cls.__openllm_sampling_class__)}
    )
    cls = _ConfigBuilder(cls, these).add_attrs_init().add_repr().build_class()

    # Finally, resolve the types
    if getattr(cls, '__attrs_types_resolved__', None) != cls:
      # NOTE: We will try to resolve type here, and cached it for faster use
      globs: DictStrAny = {'t': t, 'typing': t}
      if cls.__module__ in sys.modules:
        globs.update(sys.modules[cls.__module__].__dict__)
      attr.resolve_types(cls.__openllm_generation_class__, globalns=globs)
      attr.resolve_types(cls.__openllm_sampling_class__, globalns=globs)
      cls = attr.resolve_types(cls, globalns=globs)
    # the hint cache for easier access
    cls.__openllm_hints__ = {
      f.name: f.type
      for ite in [
        attr.fields(cls),
        attr.fields(cls.__openllm_generation_class__),
        attr.fields(cls.__openllm_sampling_class__),
      ]
      for f in ite
    }

    # for pickling to work, need to set the module to the correct outer frame
    try:
      cls.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
      pass

  def __setattr__(self, attr: str, value: t.Any) -> None:
    if attr in _reserved_namespace:
      raise ForbiddenAttributeError(
        f'{attr} should not be set during runtime as these value will be reflected during runtime. Instead, you can create a custom LLM subclass {self.__class__.__name__}.'
      )
    super().__setattr__(attr, value)

  def __init__(
    self,
    *,
    generation_config: DictStrAny | None = None,
    sampling_config: DictStrAny | None = None,
    __openllm_extras__: DictStrAny | None = None,
    __openllm_config_override__: DictStrAny | None = None,
    **attrs: t.Any,
  ):
    # create a copy of the keys as cache
    _cached_keys = tuple(attrs.keys())
    _generation_cl_dict = attr.fields_dict(self.__openllm_generation_class__)
    _sampling_cl_dict = attr.fields_dict(self.__openllm_sampling_class__)
    if generation_config is None:
      generation_config = {k: v for k, v in attrs.items() if k in _generation_cl_dict}
    else:
      generation_config = config_merger.merge(
        generation_config, {k: v for k, v in attrs.items() if k in _generation_cl_dict}
      )

    if sampling_config is None:
      sampling_config = {k: v for k, v in attrs.items() if k in _sampling_cl_dict}
    else:
      sampling_config = config_merger.merge(
        sampling_config, {k: v for k, v in attrs.items() if k in _sampling_cl_dict}
      )
    for k in _cached_keys:
      if k in generation_config or k in sampling_config or attrs[k] is None:
        del attrs[k]

    self.__openllm_config_override__ = __openllm_config_override__ or {}
    self.__openllm_extras__ = config_merger.merge(
      first_not_none(__openllm_extras__, default={}),
      {k: v for k, v in attrs.items() if k not in self.__openllm_accepted_keys__},
    )
    self.generation_config = self['generation_class'](_internal=True, **generation_config)
    self.sampling_config = self['sampling_class'].from_generation_config(self.generation_config, **sampling_config)

    # The rest of attrs should only be the attributes to be passed to __attrs_init__
    self.__attrs_init__(**attrs)

  # fmt: off
  # update-config-stubs.py: start
  # NOTE: ModelSettings arguments
  @overload
  def __getitem__(self,item:t.Literal['default_id'])->str:...
  @overload
  def __getitem__(self,item:t.Literal['model_ids'])->ListStr:...
  @overload
  def __getitem__(self,item:t.Literal['architecture'])->str:...
  @overload
  def __getitem__(self,item:t.Literal['url'])->str:...
  @overload
  def __getitem__(self,item:t.Literal['serialisation'])->LiteralSerialisation:...
  @overload
  def __getitem__(self,item:t.Literal['trust_remote_code'])->bool:...
  @overload
  def __getitem__(self,item:t.Literal['service_name'])->str:...
  @overload
  def __getitem__(self,item:t.Literal['requirements'])->t.Optional[ListStr]:...
  @overload
  def __getitem__(self,item:t.Literal['model_type'])->t.Literal['causal_lm', 'seq2seq_lm']:...
  @overload
  def __getitem__(self,item:t.Literal['name_type'])->t.Optional[t.Literal['dasherize', 'lowercase']]:...
  @overload
  def __getitem__(self,item:t.Literal['backend'])->t.Tuple[LiteralBackend, ...]:...
  @overload
  def __getitem__(self,item:t.Literal['model_name'])->str:...
  @overload
  def __getitem__(self,item:t.Literal['start_name'])->str:...
  @overload
  def __getitem__(self,item:t.Literal['timeout'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['workers_per_resource'])->t.Union[int, float]:...
  @overload
  def __getitem__(self,item:t.Literal['fine_tune_strategies'])->t.Dict[AdapterType, FineTuneConfig]:...
  @overload
  def __getitem__(self,item:t.Literal['add_generation_prompt'])->bool:...
  # NOTE: generation_class, sampling_class and extras arguments
  @overload
  def __getitem__(self,item:t.Literal['generation_class'])->t.Type[openllm_core.GenerationConfig]:...
  @overload
  def __getitem__(self,item:t.Literal['sampling_class'])->t.Type[openllm_core.SamplingParams]:...
  @overload
  def __getitem__(self,item:t.Literal['extras'])->t.Dict[str, t.Any]:...
  # NOTE: GenerationConfig arguments
  @overload
  def __getitem__(self,item:t.Literal['max_new_tokens'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['min_length'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['min_new_tokens'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['early_stopping'])->bool:...
  @overload
  def __getitem__(self,item:t.Literal['max_time'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['num_beams'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['num_beam_groups'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['penalty_alpha'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['use_cache'])->bool:...
  @overload
  def __getitem__(self,item:t.Literal['temperature'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['top_k'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['top_p'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['typical_p'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['epsilon_cutoff'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['eta_cutoff'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['diversity_penalty'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['repetition_penalty'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['encoder_repetition_penalty'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['length_penalty'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['no_repeat_ngram_size'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['bad_words_ids'])->t.List[t.List[int]]:...
  @overload
  def __getitem__(self,item:t.Literal['force_words_ids'])->t.Union[t.List[t.List[int]], t.List[t.List[t.List[int]]]]:...
  @overload
  def __getitem__(self,item:t.Literal['renormalize_logits'])->bool:...
  @overload
  def __getitem__(self,item:t.Literal['forced_bos_token_id'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['forced_eos_token_id'])->t.Union[int, t.List[int]]:...
  @overload
  def __getitem__(self,item:t.Literal['remove_invalid_values'])->bool:...
  @overload
  def __getitem__(self,item:t.Literal['exponential_decay_length_penalty'])->t.Tuple[int, float]:...
  @overload
  def __getitem__(self,item:t.Literal['suppress_tokens'])->t.List[int]:...
  @overload
  def __getitem__(self,item:t.Literal['begin_suppress_tokens'])->t.List[int]:...
  @overload
  def __getitem__(self,item:t.Literal['forced_decoder_ids'])->t.List[t.List[int]]:...
  @overload
  def __getitem__(self,item:t.Literal['num_return_sequences'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['output_attentions'])->bool:...
  @overload
  def __getitem__(self,item:t.Literal['output_hidden_states'])->bool:...
  @overload
  def __getitem__(self,item:t.Literal['output_scores'])->bool:...
  @overload
  def __getitem__(self,item:t.Literal['pad_token_id'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['bos_token_id'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['eos_token_id'])->t.Union[int, t.List[int]]:...
  @overload
  def __getitem__(self,item:t.Literal['encoder_no_repeat_ngram_size'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['decoder_start_token_id'])->int:...
  # NOTE: SamplingParams arguments
  @overload
  def __getitem__(self,item:t.Literal['n'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['best_of'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['presence_penalty'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['frequency_penalty'])->float:...
  @overload
  def __getitem__(self,item:t.Literal['use_beam_search'])->bool:...
  @overload
  def __getitem__(self,item:t.Literal['ignore_eos'])->bool:...
  @overload
  def __getitem__(self,item:t.Literal['logprobs'])->int:...
  @overload
  def __getitem__(self,item:t.Literal['prompt_logprobs'])->t.Optional[int]:...
  @overload
  def __getitem__(self,item:t.Literal['skip_special_tokens'])->bool:...
  # NOTE: PeftType arguments
  @overload
  def __getitem__(self,item:t.Literal['prompt_tuning'])->t.Dict[str, t.Any]:...
  @overload
  def __getitem__(self,item:t.Literal['multitask_prompt_tuning'])->t.Dict[str, t.Any]:...
  @overload
  def __getitem__(self,item:t.Literal['p_tuning'])->t.Dict[str, t.Any]:...
  @overload
  def __getitem__(self,item:t.Literal['prefix_tuning'])->t.Dict[str, t.Any]:...
  @overload
  def __getitem__(self,item:t.Literal['lora'])->t.Dict[str, t.Any]:...
  @overload
  def __getitem__(self,item:t.Literal['adalora'])->t.Dict[str, t.Any]:...
  @overload
  def __getitem__(self,item:t.Literal['adaption_prompt'])->t.Dict[str, t.Any]:...
  @overload
  def __getitem__(self,item:t.Literal['ia3'])->t.Dict[str, t.Any]:...
  @overload
  def __getitem__(self,item:t.Literal['loha'])->t.Dict[str, t.Any]:...
  @overload
  def __getitem__(self,item:t.Literal['lokr'])->t.Dict[str, t.Any]:...
  # update-config-stubs.py: stop
  # fmt: on

  def __getitem__(self, item: LiteralString | t.Any) -> t.Any:
    """Allowing access LLMConfig as a dictionary. The order will always evaluate as.

    __openllm_*__ > self.key > self.generation_config > self['fine_tune_strategies'] > __openllm_extras__

    This method is purely for convenience, and should not be used for performance critical code.
    """
    if item is None:
      raise TypeError(f"{self} doesn't understand how to index None.")
    item = inflection.underscore(item)
    if item in _reserved_namespace:
      raise ForbiddenAttributeError(
        f"'{item}' is a reserved namespace for {self.__class__} and should not be access nor modified."
      )
    internal_attributes = f'__openllm_{item}__'
    if hasattr(self, internal_attributes):
      if item in self.__openllm_config_override__:
        return self.__openllm_config_override__[item]
      return getattr(self, internal_attributes)
    elif hasattr(self, item):
      return getattr(self, item)
    elif hasattr(self.__openllm_generation_class__, item):
      return getattr(self.generation_config, item)
    elif hasattr(self.__openllm_sampling_class__, item):
      return getattr(self.sampling_config, item)
    elif item in self.__class__.__openllm_fine_tune_strategies__:
      return self.__class__.__openllm_fine_tune_strategies__[t.cast(AdapterType, item)]
    elif item in self.__openllm_extras__:
      return self.__openllm_extras__[item]
    else:
      raise KeyError(item)

  def __getattribute__(self, item: str) -> t.Any:
    if item in _reserved_namespace:
      raise ForbiddenAttributeError(
        f"'{item}' belongs to a private namespace for {self.__class__} and should not be access nor modified."
      )
    return _object_getattribute.__get__(self)(item)

  def __len__(self) -> int:
    return len(self.__openllm_accepted_keys__) + len(self.__openllm_extras__)

  def keys(self) -> list[str]:
    return list(self.__openllm_accepted_keys__) + list(self.__openllm_extras__)

  def values(self) -> list[t.Any]:
    return (
      [getattr(self, k.name) for k in attr.fields(self.__class__)]
      + [getattr(self.generation_config, k.name) for k in attr.fields(self.__openllm_generation_class__)]
      + [getattr(self.sampling_config, k.name) for k in attr.fields(self.__openllm_sampling_class__)]
      + list(self.__openllm_extras__.values())
    )

  def items(self) -> list[tuple[str, t.Any]]:
    return (
      [(k.name, getattr(self, k.name)) for k in attr.fields(self.__class__)]
      + [(k.name, getattr(self.generation_config, k.name)) for k in attr.fields(self.__openllm_generation_class__)]
      + [(k.name, getattr(self.sampling_config, k.name)) for k in attr.fields(self.__openllm_sampling_class__)]
      + list(self.__openllm_extras__.items())
    )

  def __iter__(self) -> t.Iterator[str]:
    return iter(self.keys())

  def __contains__(self, item: t.Any) -> bool:
    if item in self.__openllm_extras__:
      return True
    return item in self.__openllm_accepted_keys__

  @classmethod
  def model_derivate(cls, name: str | None = None, **attrs: t.Any) -> LLMConfig:
    """A helper class to generate a new LLMConfig class with additional attributes.

    This is useful to modify builtin __config__ value attributes.

    ```python
    class DollyV2Config(openllm.LLMConfig):
        ...

    my_new_class = DollyV2Config.model_derivate(default_id='...')
    ```

    Args:
        name: The name of the new class.
        **attrs: The attributes to be added to the new class. This will override
                any existing attributes with the same name.
    """
    if not hasattr(cls, '__config__'):
      raise ValueError('Cannot derivate a LLMConfig without __config__')
    _new_cfg = {k: v for k, v in attrs.items() if k in attr.fields_dict(_ModelSettingsAttr)}
    attrs = {k: v for k, v in attrs.items() if k not in _new_cfg}
    new_cls = types.new_class(
      name or f"{cls.__name__.replace('Config', '')}DerivateConfig",
      (cls,),
      {},
      lambda ns: ns.update(
        {
          '__config__': config_merger.merge(copy.deepcopy(cls.__dict__['__config__']), _new_cfg),
          '__base_config__': cls,  # keep a reference for easy access
        }
      ),
    )

    # For pickling to work, the __module__ variable needs to be set to the
    # frame where the class is created.  Bypass this step in environments where
    # sys._getframe is not defined (Jython for example) or sys._getframe is not
    # defined for arguments greater than 0 (IronPython).
    try:
      new_cls.__module__ = sys._getframe(1).f_globals.get('__name__', '__main__')
    except (AttributeError, ValueError):
      pass
    return new_cls(**attrs)

  def model_dump(self, flatten: bool = False, **_: t.Any) -> DictStrAny:
    dumped = converter.unstructure(self)
    generation_config = converter.unstructure(self.generation_config)
    sampling_config = converter.unstructure(self.sampling_config)
    if flatten:
      dumped.update(generation_config)
    else:
      dumped['generation_config'] = generation_config
    dumped.update(sampling_config)
    return dumped

  def model_dump_json(self, **kwargs: t.Any) -> bytes:
    return orjson.dumps(self.model_dump(**kwargs))

  @classmethod
  def model_construct_json(cls, json_str: str | bytes) -> Self:
    try:
      attrs = orjson.loads(json_str)
    except orjson.JSONDecodeError as err:
      raise openllm_core.exceptions.ValidationError(f'Failed to load JSON: {err}') from None
    return converter.structure(attrs, cls)

  @classmethod
  def model_construct_env(cls, **attrs: t.Any) -> Self:
    """A helpers that respect configuration values environment variables."""
    attrs = {k: v for k, v in attrs.items() if v is not None}
    env_json_string = os.environ.get('OPENLLM_CONFIG', None)

    config_from_env: DictStrAny = {}
    if env_json_string is not None:
      try:
        config_from_env = orjson.loads(env_json_string)
      except orjson.JSONDecodeError as e:
        raise RuntimeError("Failed to parse 'OPENLLM_CONFIG' as valid JSON string.") from e

    if 'generation_config' in attrs and 'sampling_config' in attrs:  # backward compatibility
      generation_config = attrs.pop('generation_config')
      sampling_config = attrs.pop('sampling_config')
    elif 'llm_config' in attrs:  # NOTE: this is the new key
      llm_config = attrs.pop('llm_config')
      generation_config = {
        k: v for k, v in llm_config.items() if k in attr.fields_dict(cls.__openllm_generation_class__)
      }
      sampling_config = {k: v for k, v in llm_config.items() if k in attr.fields_dict(cls.__openllm_sampling_class__)}
    else:
      generation_config = {k: v for k, v in attrs.items() if k in attr.fields_dict(cls.__openllm_generation_class__)}
      sampling_config = {k: v for k, v in attrs.items() if k in attr.fields_dict(cls.__openllm_sampling_class__)}

    for k in tuple(attrs.keys()):
      if k in generation_config or k in sampling_config:
        del attrs[k]

    config_from_env.update(attrs)
    config_from_env.update(sampling_config)
    config_from_env.update(generation_config)
    return converter.structure(config_from_env, cls)

  def model_validate_click(self, **attrs: t.Any) -> tuple[LLMConfig, DictStrAny]:
    """Parse given click attributes into a LLMConfig and return the remaining click attributes."""
    llm_config_attrs: DictStrAny = {'generation_config': {}, 'sampling_config': {}}
    key_to_remove: ListStr = []
    for k, v in attrs.items():
      if k.startswith(f"{self['model_name']}_generation_"):  # NOTE: This is an internal state for openllm cli.
        llm_config_attrs['generation_config'][k[len(self['model_name'] + '_generation_') :]] = v
        key_to_remove.append(k)
      elif k.startswith('_openllm_genericinternal_generation_'):
        llm_config_attrs['generation_config'][k[len('_openllm_genericinternal_generation_') :]] = v
        key_to_remove.append(k)
      elif k.startswith(f"{self['model_name']}_sampling_"):
        llm_config_attrs['sampling_config'][k[len(self['model_name'] + '_sampling_') :]] = v
        key_to_remove.append(k)
      elif k.startswith('_openllm_genericinternal_sampling_'):
        llm_config_attrs['sampling_config'][k[len('_openllm_genericinternal_sampling_') :]] = v
        key_to_remove.append(k)
      elif k.startswith(f"{self['model_name']}_"):
        llm_config_attrs[k[len(self['model_name'] + '_') :]] = v
        key_to_remove.append(k)
    return self.model_construct_env(**llm_config_attrs), {k: v for k, v in attrs.items() if k not in key_to_remove}

  def make_fine_tune_config(self, adapter_type: AdapterType, **attrs: t.Any) -> FineTuneConfig:
    return FineTuneConfig(adapter_type=adapter_type, llm_config_class=self.__class__).with_config(**attrs)

  @overload
  def to_generation_config(self, return_as_dict: t.Literal[False] = False) -> transformers.GenerationConfig: ...

  @overload
  def to_generation_config(self, return_as_dict: t.Literal[True] = ...) -> DictStrAny: ...

  def to_generation_config(self, return_as_dict: bool = False) -> transformers.GenerationConfig | DictStrAny:
    config = transformers.GenerationConfig(**converter.unstructure(self.generation_config))
    return config.to_dict() if return_as_dict else config

  def to_sampling_config(self) -> vllm.SamplingParams:
    return self.sampling_config.build()

  @overload
  def with_request(self, request: ChatCompletionRequest | CompletionRequest) -> dict[str, t.Any]: ...

  @overload
  def with_request(self, request: CohereChatRequest | CohereGenerateRequest) -> dict[str, t.Any]: ...

  def with_request(self, request: AttrsInstance) -> dict[str, t.Any]:
    if importlib.util.find_spec('openllm') is None:
      raise MissingDependencyError(
        "'openllm' is required to use 'with_request'. Make sure to install with 'pip install openllm'."
      )
    from openllm.protocol.cohere import CohereChatRequest, CohereGenerateRequest
    from openllm.protocol.openai import ChatCompletionRequest, CompletionRequest

    if isinstance(request, (ChatCompletionRequest, CompletionRequest)):
      return self._with_openai_request(request)
    elif isinstance(request, (CohereChatRequest, CohereGenerateRequest)):
      return self._with_cohere_request(request)
    else:
      raise TypeError(f'Unknown request type {type(request)}')

  def _with_openai_request(self, request: ChatCompletionRequest | CompletionRequest) -> dict[str, t.Any]:
    d = dict(
      temperature=first_not_none(request.temperature, self['temperature']),
      top_p=first_not_none(request.top_p, self['top_p']),
      top_k=first_not_none(request.top_k, self['top_k']),
      best_of=first_not_none(request.best_of, self['best_of']),
      n=first_not_none(request.n, default=self['n']),
      stop=first_not_none(request.stop, default=None),
      max_new_tokens=first_not_none(request.max_tokens, default=self['max_new_tokens']),
      presence_penalty=first_not_none(request.presence_penalty, default=self['presence_penalty']),
      frequency_penalty=first_not_none(request.frequency_penalty, default=self['frequency_penalty']),
    )
    if hasattr(request, 'logprobs'):
      d['logprobs'] = first_not_none(request.logprobs, default=self['logprobs'])
    return d

  def _with_cohere_request(self, request: CohereGenerateRequest | CohereChatRequest) -> dict[str, t.Any]:
    d = dict(
      max_new_tokens=first_not_none(request.max_tokens, default=self['max_new_tokens']),
      temperature=first_not_none(request.temperature, default=self['temperature']),
      top_k=first_not_none(request.k, default=self['top_k']),
      top_p=first_not_none(request.p, default=self['top_p']),
    )
    if hasattr(request, 'num_generations'):
      d['n'] = first_not_none(request.num_generations, default=self['n'])
    if hasattr(request, 'frequency_penalty'):
      d['frequency_penalty'] = first_not_none(request.frequency_penalty, default=self['frequency_penalty'])
    if hasattr(request, 'presence_penalty'):
      d['presence_penalty'] = first_not_none(request.presence_penalty, default=self['presence_penalty'])
    return d

  @classmethod
  def to_click_options(cls, f: AnyCallable) -> click.Command:
    """Convert current configuration to click options.

    This can be used as a decorator for click commands.

    > [!NOTE]
    > The identifier for all LLMConfig will be prefixed with '<model_name>_*', and the generation config will be prefixed with '<model_name>_generation_*'.
    """
    for name, field in attr.fields_dict(cls.__openllm_generation_class__).items():
      ty = cls.__openllm_hints__.get(name)
      # NOTE: Union type is currently not yet supported, we probably just need to use environment instead.
      if t.get_origin(ty) is t.Union:
        continue
      f = dantic.attrs_to_options(name, field, cls.__openllm_model_name__, typ=ty, suffix_generation=True)(f)
    f = cog.optgroup.group('GenerationConfig generation options')(f)

    for name, field in attr.fields_dict(cls.__openllm_sampling_class__).items():
      ty = cls.__openllm_hints__.get(name)
      # NOTE: Union type is currently not yet supported, we probably just need to use environment instead.
      if t.get_origin(ty) is t.Union:
        continue
      f = dantic.attrs_to_options(name, field, cls.__openllm_model_name__, typ=ty, suffix_sampling=True)(f)
    f = cog.optgroup.group('SamplingParams sampling options')(f)

    total_keys = set(attr.fields_dict(cls.__openllm_generation_class__)) | set(
      attr.fields_dict(cls.__openllm_sampling_class__)
    )

    if len(cls.__openllm_accepted_keys__.difference(total_keys)) == 0:
      return t.cast('click.Command', f)
    # We pop out 'generation_config' as it is a attribute that we don't need to expose to CLI.
    for name, field in attr.fields_dict(cls).items():
      ty = cls.__openllm_hints__.get(name)
      # NOTE: Union type is currently not yet supported, we probably just need to use environment instead.
      if t.get_origin(ty) is t.Union or name == 'generation_config' or name == 'sampling_config':
        continue
      f = dantic.attrs_to_options(name, field, cls.__openllm_model_name__, typ=ty)(f)

    return cog.optgroup.group(f'{cls.__name__} options')(f)

  # holds a mapping from self.__openllm_model_type__ to peft.TaskType
  @classmethod
  def peft_task_type(cls) -> str:
    return PEFT_TASK_TYPE_TARGET_MAPPING[cls.__openllm_model_type__]


converter.register_unstructure_hook_factory(
  lambda cls: lenient_issubclass(cls, LLMConfig),
  lambda cls: make_dict_unstructure_fn(cls, converter, _cattrs_omit_if_default=False, _cattrs_use_linecache=True),
)


def structure_llm_config(data: t.Any, cls: type[LLMConfig]) -> LLMConfig:
  if not isinstance(data, dict):
    raise RuntimeError(f'Expected a dictionary, but got {type(data)}')
  _config_override = {k: v for k, v in data.items() if k in cls.__config__}
  cls_attrs = {k: v for k, v in data.items() if k in cls.__openllm_accepted_keys__}
  generation_cls_fields = attr.fields_dict(cls.__openllm_generation_class__)
  sampling_cls_fields = attr.fields_dict(cls.__openllm_sampling_class__)
  if 'generation_config' in data:
    generation_config = data.pop('generation_config')
    if not isinstance(generation_config, dict):
      raise RuntimeError(f'Expected a dictionary, but got {type(generation_config)}')
    config_merger.merge(generation_config, {k: v for k, v in data.items() if k in generation_cls_fields})
  else:
    generation_config = {k: v for k, v in data.items() if k in generation_cls_fields}
  if 'sampling_config' in data:
    sampling_config = data.pop('sampling_config')
    if not isinstance(sampling_config, dict):
      raise RuntimeError(f'Expected a dictionary, but got {type(sampling_config)}')
    config_merger.merge(sampling_config, {k: v for k, v in data.items() if k in sampling_cls_fields})
  else:
    sampling_config = {k: v for k, v in data.items() if k in sampling_cls_fields}
  # The rest should be passed to extras
  data = {k: v for k, v in data.items() if k not in cls.__openllm_accepted_keys__ and k not in _config_override}
  return cls(
    generation_config=generation_config,
    sampling_config=sampling_config,
    __openllm_extras__=data,
    __openllm_config_override__=_config_override,
    **cls_attrs,
  )


converter.register_structure_hook_func(lambda cls: lenient_issubclass(cls, LLMConfig), structure_llm_config)
openllm_home = os.path.expanduser(
  os.environ.get(
    'OPENLLM_HOME',
    os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache')), 'openllm'),
  )
)
