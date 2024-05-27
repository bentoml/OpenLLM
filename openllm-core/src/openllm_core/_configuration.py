from __future__ import annotations
import abc, inspect, logging, os, typing as t
import inflection, orjson, pydantic
from deepmerge.merger import Merger

from ._typing_compat import (
  DictStrAny,
  ListStr,
  LiteralSerialisation,
  NotRequired,
  Required,
  Self,
  TypedDict,
  overload,
  Annotated,
)
from .exceptions import ForbiddenAttributeError, MissingDependencyError
from .utils import field_env_key, is_vllm_available, is_transformers_available

if t.TYPE_CHECKING:
  import transformers, vllm, openllm, torch

  from ._schemas import MessageParam

__all__ = ['GenerationConfig', 'LLMConfig', 'field_env_key']

logger = logging.getLogger(__name__)
config_merger = Merger([(dict, 'merge')], ['override'], ['override'])
_object_setattr = object.__setattr__


# NOTE: This is actually torch tensor, but we don't want to import torch here.
if t.TYPE_CHECKING:
  Tensor = torch.Tensor
else:
  Tensor = t.Any

LogitsProcessor = t.Callable[[t.List[int], Tensor], Tensor]


class GenerationConfig(pydantic.BaseModel):
  model_config = pydantic.ConfigDict(extra='allow')

  min_length: int = pydantic.Field(
    0,
    ge=0,  #
    description='The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.',
  )
  min_new_tokens: t.Optional[int] = pydantic.Field(
    None, description='The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.'
  )
  max_time: t.Optional[float] = pydantic.Field(
    None,
    description='The maximum amount of time you allow the computation to run for in seconds. generation will still finish the current pass after allocated time has been passed.',
  )
  num_beams: int = pydantic.Field(1, description='Number of beams for beam search. 1 means no beam search.')
  num_beam_groups: int = pydantic.Field(
    1,
    description='Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.',
  )
  penalty_alpha: t.Optional[float] = pydantic.Field(
    None,
    description='The values balance the model confidence and the degeneration penalty in contrastive search decoding.',
  )
  use_cache: bool = pydantic.Field(
    True,
    description='Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.',
  )
  typical_p: float = pydantic.Field(
    1.0,
    description='Local typicality measures how similar the conditional probability of predicting a target token next is to the expected conditional probability of predicting a random token next, given the partial text already generated. If set to float < 1, the smallest set of the most locally typical tokens with probabilities that add up to `typical_p` or higher are kept for generation. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.',
  )
  epsilon_cutoff: float = pydantic.Field(
    0.0,
    description='If set to float strictly between 0 and 1, only tokens with a conditional probability greater than `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more details.',
  )
  eta_cutoff: float = pydantic.Field(
    0.0,
    description='Eta sampling is a hybrid of locally typical sampling and epsilon sampling. If set to float strictly between 0 and 1, a token is only considered if it is greater than either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`. The latter term is intuitively the expected next token probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested values range from 3e-4 to 2e-3, depending on the size of the model. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more details. ',
  )
  diversity_penalty: float = pydantic.Field(
    0.0,
    description="This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled. ",
  )
  repetition_penalty: float = pydantic.Field(
    1.0,
    description='The parameter for repetition penalty. 1.0 means no penalty. See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.',
  )
  encoder_repetition_penalty: float = pydantic.Field(
    1.0,
    description='The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are not in the original input. 1.0 means no penalty.',
  )
  no_repeat_ngram_size: int = pydantic.Field(
    0, description='If set to int > 0, all ngrams of that size can only occur once.'
  )
  bad_words_ids: t.Optional[t.List[t.List[int]]] = pydantic.Field(
    None,
    description='List of token ids that are not allowed to be generated. In order to get the token ids of the words that should not appear in the generated text, use `tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids`.',
  )
  force_words_ids: t.Optional[t.Union[t.List[t.List[int]], t.List[t.List[t.List[int]]]]] = pydantic.Field(
    None,
    description='List of token ids that must be generated. If given a `List[List[int]]`, this is treated as a simple list of words that must be included, the opposite to `bad_words_ids`. If given `List[List[List[int]]]`, this triggers a [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one can allow different forms of each word. ',
  )
  renormalize_logits: bool = pydantic.Field(
    False,
    description="Whether to renormalize the logits after applying all the logits processors or warpers (including the custom ones). It's highly recommended to set this flag to `True` as the search algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization. ",
  )
  forced_bos_token_id: t.Optional[int] = pydantic.Field(
    None,
    description='The id of the token to force as the first generated token after the ``decoder_start_token_id``. Useful for multilingual models like [mBART](https://huggingface.co/docs/transformers/model_doc/mbart) where the first generated token needs to be the target language token. ',
  )
  forced_eos_token_id: t.Optional[t.Union[int, t.List[int]]] = pydantic.Field(
    None,
    description='The id of the token to force as the last generated token when `max_length` is reached. Optionally, use a list to set multiple *end-of-sequence* tokens.',
  )
  remove_invalid_values: bool = pydantic.Field(
    False,
    description='Whether to remove possible *nan* and *inf* outputs of the model to prevent the generation method to crash. Note that using `remove_invalid_values` can slow down generation.',
  )
  exponential_decay_length_penalty: t.Optional[t.Tuple[int, float]] = pydantic.Field(
    None,
    description='This tuple adds an exponentially increasing length penalty, after a certain amount of tokens have been generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` indicates where penalty starts and `decay_factor` represents the factor of exponential decay',
  )
  suppress_tokens: t.Optional[t.List[int]] = pydantic.Field(
    None,
    description='A list of tokens that will be suppressed at generation. The `SupressTokens` logit processor will set their log probs to `-inf` so that they are not sampled.',
  )
  begin_suppress_tokens: t.Optional[t.List[int]] = pydantic.Field(
    None,
    description='A list of tokens that will be suppressed at the beginning of the generation. The `SupressBeginTokens` logit processor will set their log probs to `-inf` so that they are not sampled. ',
  )
  forced_decoder_ids: t.Optional[t.List[t.List[int]]] = pydantic.Field(
    None,
    description='A list of pairs of integers which indicates a mapping from generation indices to token indices that will be forced before sampling. For example, `[[1, 123]]` means the second generated token will always be a token of index 123.',
  )
  num_return_sequences: int = pydantic.Field(
    1, description='The number of independently computed returned sequences for each element in the batch.'
  )
  output_attentions: bool = pydantic.Field(
    False,
    description='Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more details.',
  )
  output_hidden_states: bool = pydantic.Field(
    False,
    description='Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more details.',
  )
  output_scores: bool = pydantic.Field(
    False,
    description='Whether or not to return the prediction scores. See `scores` under returned tensors for more details.',
  )
  pad_token_id: t.Optional[int] = pydantic.Field(None, description='The id of the *padding* token.')
  bos_token_id: t.Optional[int] = pydantic.Field(None, description='The id of the *beginning-of-sequence* token.')
  eos_token_id: t.Optional[t.Union[int, t.List[int]]] = pydantic.Field(
    None,
    description='The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.',
  )
  encoder_no_repeat_ngram_size: int = pydantic.Field(
    0,
    description='If set to int > 0, all ngrams of that size that occur in the `encoder_input_ids` cannot occur in the `decoder_input_ids`.',
  )
  decoder_start_token_id: int = pydantic.Field(
    None,
    description='If an encoder-decoder model starts decoding with a different token than *bos*, the id of that token.',
  )
  # NOTE vLLM compatible fields.
  n: int = pydantic.Field(1, description='Number of output sequences to return for the given prompt.')
  best_of: t.Optional[int] = pydantic.Field(
    None,
    description='Number of output sequences that are generated from the prompt. From these `best_of` sequences, the top `n` sequences are returned. `best_of` must be greater than or equal to `n`. This is treated as the beam width when `use_beam_search` is True. By default, `best_of` is set to `n`.',
  )
  presence_penalty: float = pydantic.Field(
    0.0,
    description='Float that penalizes new tokens based on whether they appear in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.',
  )
  frequency_penalty: float = pydantic.Field(
    0.0,
    description='Float that penalizes new tokens based on their frequency in the generated text so far. Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens.',
  )
  temperature: float = pydantic.Field(
    1.0, ge=0.0, le=1.0, description='The value used to modulate the next token probabilities.'
  )
  top_k: int = pydantic.Field(
    50, description='The number of highest probability vocabulary tokens to keep for top-k-filtering.'
  )
  top_p: float = pydantic.Field(
    1.0,
    description='If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.',
  )
  min_p: float = pydantic.Field(
    0.0,
    ge=0,
    le=1.0,
    description='Float that represents the minimum probability for a token to be considered, relative to the probability of the most likely token. Must be in [0, 1]. Set to 0 to disable this.',
  )
  use_beam_search: bool = pydantic.Field(False, description='Whether to use beam search instead of sampling.')
  length_penalty: float = pydantic.Field(
    1.0,
    description='Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while `length_penalty` < 0.0 encourages shorter sequences.',
  )
  early_stopping: bool = pydantic.Field(
    False,
    description="Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values: `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; `'never'`, where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm) ",
  )
  stop: t.Optional[t.Union[str, t.List[str]]] = pydantic.Field(
    None,
    description='List of strings that stop the generation when they are generated. The returned output will not contain the stop strings.',
  )
  stop_token_ids: t.Optional[t.List[int]] = pydantic.Field(
    None,
    description='List of tokens that stop the generation when they are generated. The returned output will contain the stop tokens unless the stop tokens are special tokens.',
  )
  include_stop_str_in_output: bool = pydantic.Field(
    False, description='Whether to include the stop strings in output text. Defaults to False.'
  )
  ignore_eos: bool = pydantic.Field(
    False,
    description='Whether to ignore the EOS token and continue generating tokens after the EOS token is generated.',
  )
  max_tokens: int = pydantic.Field(
    20,
    ge=0,
    description='The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.',
    alias='max_new_tokens',
  )
  min_tokens: int = pydantic.Field(
    0,
    ge=0,
    description='Minimum number of tokens to generate per output sequence before EOS or stop_token_ids can be generated',
    alias='min_new_tokens',
  )
  logprobs: t.Optional[int] = pydantic.Field(
    None, description='Number of log probabilities to return per output token.'
  )
  detokenize: bool = pydantic.Field(True, description='Whether to detokenize the output.')
  truncate_prompt_tokens: t.Optional[Annotated[int, pydantic.Field(ge=1)]] = pydantic.Field(
    None, description='Truncate the prompt tokens.'
  )
  prompt_logprobs: t.Optional[int] = pydantic.Field(
    None, description='Number of log probabilities to return per input token.'
  )
  skip_special_tokens: bool = pydantic.Field(
    True, description='Whether to skip special tokens in the generated output.'
  )
  spaces_between_special_tokens: bool = pydantic.Field(
    True, description='Whether to add a space between special tokens in the generated output.'
  )
  logits_processors: t.Optional[t.List[LogitsProcessor]] = pydantic.Field(
    None, description='List of functions that modify logits based on previously generated tokens.'
  )
  seed: t.Optional[int] = pydantic.Field(None, description='Random seed for generation.')

  def __getitem__(self, item: str) -> t.Any:
    if hasattr(self, item):
      return getattr(self, item)
    raise KeyError(f"'{self.__class__.__name__}' has no attribute {item}.")

  def keys(self):
    key = list(self.model_fields.keys())
    if self.model_extra:
      key += list(self.model_extra.keys())
    return key

  def values(self):
    return [getattr(self, k) for k in self.keys()]

  def items(self):
    return [(k, getattr(self, k)) for k in self.keys()]

  def build(self, config: t.Literal['vllm', 'pt']) -> t.Any:
    if config == 'vllm':
      if not is_vllm_available():
        raise MissingDependencyError(
          'vLLM is not installed. Make sure to install it with `pip install "openllm-core[vllm]"`'
        )
      from vllm import SamplingParams

      return SamplingParams(**{k: getattr(self, k) for k in set(inspect.signature(SamplingParams).parameters.keys())})
    elif config == 'pt':
      if not is_transformers_available():
        raise MissingDependencyError(
          'transformers is not installed. Make sure to install it with `pip install "openllm-core[transformers]"`'
        )
      import transformers

      return transformers.GenerationConfig(max_new_tokens=self.max_tokens, **{k: v for k, v in self.items()})
    raise ValueError(f'Unknown config type: {config}')


# cached it here to save one lookup per assignment
_object_getattribute = object.__getattribute__


class ModelSettings(TypedDict, total=False):
  default_id: Required[str]
  model_ids: Required[ListStr]
  architecture: Required[str]
  url: str
  serialisation: LiteralSerialisation
  trust_remote_code: bool
  service_name: NotRequired[str]
  requirements: t.Optional[ListStr]
  # llm implementation specifics
  model_type: t.Literal['causal_lm', 'seq2seq_lm']
  timeout: int

  # the target generation_config class to be used.
  fine_tune_strategies: t.Tuple[t.Dict[str, t.Any], ...]


_reserved_namespace = {'metadata_config'}

_DEFAULT = ModelSettings(
  url='',  #
  timeout=int(36e6),
  service_name='generated_service.py',  #
  model_type='causal_lm',
  requirements=None,  #
  trust_remote_code=False,
  default_id='__default__',
  model_ids=['__default__'],  #
  architecture='PreTrainedModel',
  serialisation='legacy',  #
)


class LLMConfig(pydantic.BaseModel, abc.ABC):
  model_config = pydantic.ConfigDict(extra='forbid', protected_namespaces=())

  if t.TYPE_CHECKING:
    metadata_config: ModelSettings = pydantic.Field(default_factory=dict)
    generation_config: GenerationConfig = pydantic.Field(default_factory=lambda: GenerationConfig.model_construct())

  _done_initialisation = False

  def __setattr__(self, attr: str, value: t.Any) -> None:
    if attr in _reserved_namespace and self._done_initialisation:
      raise ForbiddenAttributeError(
        f'{attr} should not be set during runtime as these value will be reflected during runtime. Instead, you can create a custom LLM subclass {self.__class__.__name__}.'
      )
    super().__setattr__(attr, value)

  @classmethod
  def __pydantic_init_subclass__(cls, **_: t.Any):
    if any(i not in cls.model_fields for i in ('metadata_config', 'generation_config')):
      raise TypeError(f'{cls.__name__} must have a `metadata_config` annd `generation_config` attribute.')

  def model_post_init(self, *_: t.Any):
    _DEFAULT.update(self.metadata_config)
    self.metadata_config = _DEFAULT
    self._done_initialisation = True

  # fmt: off
  # update-config-stubs.py: start
  # NOTE: ModelSettings arguments
  @overload
  def __getitem__(self, item: t.Literal['default_id']) -> str: ...
  @overload
  def __getitem__(self, item: t.Literal['model_ids']) -> ListStr: ...
  @overload
  def __getitem__(self, item: t.Literal['architecture']) -> str: ...
  @overload
  def __getitem__(self, item: t.Literal['url']) -> str: ...
  @overload
  def __getitem__(self, item: t.Literal['serialisation']) -> LiteralSerialisation: ...
  @overload
  def __getitem__(self, item: t.Literal['trust_remote_code']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['service_name']) -> str: ...
  @overload
  def __getitem__(self, item: t.Literal['requirements']) -> t.Optional[ListStr]: ...
  @overload
  def __getitem__(self, item: t.Literal['model_type']) -> t.Literal['causal_lm', 'seq2seq_lm']: ...
  @overload
  def __getitem__(self, item: t.Literal['timeout']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['fine_tune_strategies']) -> t.Tuple[t.Dict[str, t.Any], ...]: ...
  # NOTE: GenerationConfig arguments
  @overload
  def __getitem__(self, item: t.Literal['min_length']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['min_new_tokens']) -> t.Optional[int]: ...
  @overload
  def __getitem__(self, item: t.Literal['max_time']) -> t.Optional[float]: ...
  @overload
  def __getitem__(self, item: t.Literal['num_beams']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['num_beam_groups']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['penalty_alpha']) -> t.Optional[float]: ...
  @overload
  def __getitem__(self, item: t.Literal['use_cache']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['typical_p']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['epsilon_cutoff']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['eta_cutoff']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['diversity_penalty']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['repetition_penalty']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['encoder_repetition_penalty']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['no_repeat_ngram_size']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['bad_words_ids']) -> t.Optional[t.List[t.List[int]]]: ...
  @overload
  def __getitem__(self, item: t.Literal['force_words_ids']) -> t.Optional[t.Union[t.List[t.List[int]], t.List[t.List[t.List[int]]]]]: ...
  @overload
  def __getitem__(self, item: t.Literal['renormalize_logits']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['forced_bos_token_id']) -> t.Optional[int]: ...
  @overload
  def __getitem__(self, item: t.Literal['forced_eos_token_id']) -> t.Optional[t.Union[int, t.List[int]]]: ...
  @overload
  def __getitem__(self, item: t.Literal['remove_invalid_values']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['exponential_decay_length_penalty']) -> t.Optional[t.Tuple[int, float]]: ...
  @overload
  def __getitem__(self, item: t.Literal['suppress_tokens']) -> t.Optional[t.List[int]]: ...
  @overload
  def __getitem__(self, item: t.Literal['begin_suppress_tokens']) -> t.Optional[t.List[int]]: ...
  @overload
  def __getitem__(self, item: t.Literal['forced_decoder_ids']) -> t.Optional[t.List[t.List[int]]]: ...
  @overload
  def __getitem__(self, item: t.Literal['num_return_sequences']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['output_attentions']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['output_hidden_states']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['output_scores']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['pad_token_id']) -> t.Optional[int]: ...
  @overload
  def __getitem__(self, item: t.Literal['bos_token_id']) -> t.Optional[int]: ...
  @overload
  def __getitem__(self, item: t.Literal['eos_token_id']) -> t.Optional[t.Union[int, t.List[int]]]: ...
  @overload
  def __getitem__(self, item: t.Literal['encoder_no_repeat_ngram_size']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['decoder_start_token_id']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['n']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['best_of']) -> t.Optional[int]: ...
  @overload
  def __getitem__(self, item: t.Literal['presence_penalty']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['frequency_penalty']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['temperature']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['top_k']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['top_p']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['min_p']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['use_beam_search']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['length_penalty']) -> float: ...
  @overload
  def __getitem__(self, item: t.Literal['early_stopping']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['stop']) -> t.Optional[t.Union[str, t.List[str]]]: ...
  @overload
  def __getitem__(self, item: t.Literal['stop_token_ids']) -> t.Optional[t.List[int]]: ...
  @overload
  def __getitem__(self, item: t.Literal['include_stop_str_in_output']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['ignore_eos']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['max_tokens']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['min_tokens']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['logprobs']) -> t.Optional[int]: ...
  @overload
  def __getitem__(self, item: t.Literal['detokenize']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['truncate_prompt_tokens']) -> t.Optional[Annotated[int, pydantic.Field(ge=1)]]: ...
  @overload
  def __getitem__(self, item: t.Literal['prompt_logprobs']) -> t.Optional[int]: ...
  @overload
  def __getitem__(self, item: t.Literal['skip_special_tokens']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['spaces_between_special_tokens']) -> bool: ...
  @overload
  def __getitem__(self, item: t.Literal['logits_processors']) -> t.Optional[t.List[LogitsProcessor]]: ...
  @overload
  def __getitem__(self, item: t.Literal['seed']) -> t.Optional[int]: ...
  @overload
  def __getitem__(self, item: t.Literal['max_new_tokens']) -> int: ...
  @overload
  def __getitem__(self, item: t.Literal['start_name']) -> str: ...
  @overload
  def __getitem__(self, item: t.Literal['model_name']) -> str: ...
  # update-config-stubs.py: stop
  # fmt: on
  def __getitem__(self, item: t.Any) -> t.Any:
    if item is None:
      raise TypeError(f"{self} doesn't understand how to index None.")
    item = inflection.underscore(item)
    if item in _reserved_namespace:
      raise ForbiddenAttributeError(
        f"'{item}' is a reserved namespace for {self.__class__} and should not be access nor modified."
      )

    # backward compatible
    if item == 'max_new_tokens':
      item = 'max_tokens'

    if self.model_extra and item in self.model_extra:
      return self.model_extra[item]
    elif hasattr(self.generation_config, item):
      return getattr(self.generation_config, item)
    elif item in self.metadata_config:
      return self.metadata_config[item]
    elif hasattr(self, item):
      return getattr(self, item)
    elif item in {'start_name', 'model_name'}:  # backward compatible
      from .config.configuration_auto import CONFIG_TO_ALIAS_NAMES

      if (cls_name := self.__class__.__name__) in CONFIG_TO_ALIAS_NAMES:
        return CONFIG_TO_ALIAS_NAMES[cls_name]

    raise KeyError(item)

  def __contains__(self, item: t.Any) -> bool:
    try:
      self[item]
      return True
    except KeyError:
      return False

  @pydantic.model_serializer
  def ser_model(self) -> dict[str, t.Any]:
    return self.generation_config.model_dump()

  @classmethod
  def model_construct_env(cls, **attrs: t.Any) -> Self:  # All LLMConfig init should start from here.
    env_json_string = os.environ.get('OPENLLM_CONFIG', None)

    config_from_env: DictStrAny = {}
    if env_json_string is not None:
      try:
        config_from_env = orjson.loads(env_json_string)
      except orjson.JSONDecodeError as e:
        raise RuntimeError("Failed to parse 'OPENLLM_CONFIG' as valid JSON string.") from e

    generation_config = {}
    if 'generation_config' in attrs and 'sampling_config' in attrs:  # backward compatibility
      generation_config = attrs.pop('generation_config')
      sampling_config = attrs.pop('sampling_config')
      generation_config.update(sampling_config)
    elif 'llm_config' in attrs:  # NOTE: this is the new key
      generation_config = attrs.pop('llm_config')

    config_from_env.update({**generation_config, **cls().generation_config.model_dump(), **attrs})
    config_from_env = {k: v for k, v in config_from_env.items() if v is not None}

    return cls.model_construct(generation_config=GenerationConfig.model_construct(**config_from_env))

  def inference_options(self, llm: openllm.LLM, backend: str | None = None) -> tuple[Self, t.Any]:
    backend = backend if backend is not None else llm.__llm_backend__
    framework = getattr(self, backend, None)
    if framework is None:
      raise ValueError(f'Unknown backend {backend}')
    try:
      return self, framework.build(self)
    except AttributeError:
      raise RuntimeError(f'Unknown backend {backend}') from None

  class vllm:
    @staticmethod
    def build(config: LLMConfig) -> vllm.SamplingParams:
      top_p = 1.0 if config['temperature'] <= 1e-5 else config['top_p']
      generation_config = config.generation_config.model_copy(update={'top_p': top_p})
      return generation_config.build('vllm')

  class pt:
    @staticmethod
    def build(config: LLMConfig) -> LLMConfig:
      return config.generation_config.build('pt')

  class hf:
    @staticmethod
    def build(config: LLMConfig) -> transformers.GenerationConfig:
      return config.generation_config.build('pt')

  @property
  def template(self) -> str:
    return '{system_message}{instruction}'

  @property
  def system_message(self) -> str:
    return ''

  @property
  def chat_template(self) -> str | None:
    return

  @property
  def chat_messages(self) -> list[MessageParam]:
    from ._schemas import MessageParam

    return [
      MessageParam(role='system', content='You are a helpful assistant'),
      MessageParam(role='user', content="Hello, I'm looking for a chatbot that can help me with my work."),
      MessageParam(role='assistant', content='Yes? What can I help you with?'),
    ]


OPENLLM_HOME = os.path.expanduser(
  os.environ.get(
    'OPENLLM_HOME',
    os.path.join(os.environ.get('XDG_CACHE_HOME', os.path.join(os.path.expanduser('~'), '.cache')), 'openllm'),
  )
)
