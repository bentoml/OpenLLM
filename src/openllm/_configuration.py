# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Configuration utilities for OpenLLM. All model configuration will inherit from openllm.configuration_utils.LLMConfig.

Note that ``openllm.LLMConfig`` is a subclass of ``pydantic.BaseModel``. It also 
has a ``to_click_options`` that returns a list of Click-compatible options for the model. 
Such options will then be parsed to ``openllm.__main__.cli``.

Each fields in ``openllm.LLMConfig`` will also automatically generate a environment
variable based on its name field.

For example, the following config class:

```python
class FlanT5Config(openllm.LLMConfig):
    custom_prompt: str = DEFAULT_PROMPT_TEMPLATE

    class GenerationConfig:
        temperature: float = 0.75
        max_length: int = 3000
        top_k: int = 50
        top_p: float = 0.4
        repetition_penalty = 1.0
```


"""
from __future__ import annotations

import copy
import os
import types
import typing as t
from abc import ABC

import click
import inflection
import orjson
import pydantic
import yaml
from bentoml._internal.models.model import ModelSignature
from click_option_group import optgroup

import openllm

from .utils import _object_setattr
from .utils.dantic import allows_multiple, parse_default

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")

    F = t.Callable[P, t.Any]

    ReprArgs: t.TypeAlias = t.Iterable[tuple[str | None, t.Any]]

    import transformers
    from pydantic.fields import FieldInfo
    from transformers.generation.beam_constraints import Constraint
else:
    from transformers.utils.dummy_pt_objects import Constraint

    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")

__all__ = ["LLMConfig", "ModelSignature"]


def field_to_options(
    name: str, field: FieldInfo, model_name: str, suffix_generation: bool = False
) -> t.Callable[[F[P]], F[P]]:
    # TODO: support parsing nested model in FieldInfo
    envvar = field.json_schema_extra.get("env") if field.json_schema_extra else None
    dasherized = inflection.dasherize(name)
    underscored = inflection.underscore(name)
    full_option_name = f"--{dasherized}"
    if field.annotation is bool:
        full_option_name += f"/--no-{dasherized}"
    if suffix_generation:
        identifier = f"{model_name}_generation_{underscored}"
    else:
        identifier = f"{model_name}_{underscored}"

    return optgroup.option(
        identifier,
        full_option_name,
        type=field.annotation,
        required=field.is_required(),
        default=parse_default(field.default, field.annotation),
        show_default=False,
        multiple=allows_multiple(field.annotation),
        help=field.description,
        show_envvar=True if envvar else False,
        envvar=envvar,
    )


class GenerationConfig(pydantic.BaseModel):
    """Generation config provides the configuration to then be parsed to ``transformers.GenerationConfig``,
    with some additional validation and environment constructor.

    Note that we always set `do_sample=True` and `return_dict_in_generate=False`
    """

    # NOTE: parameters for controlling the length of the output
    max_length: t.Optional[int] = pydantic.Field(
        20,
        description="""The maximum length the generated tokens can have. Corresponds to the length of the 
        input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.""",
    )
    max_new_tokens: t.Optional[int] = pydantic.Field(
        None, description="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."
    )
    min_length: int = pydantic.Field(
        0,
        ge=0,
        description="""The minimum length of the sequence to be generated. Corresponds to the length of the 
        input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.""",
    )
    min_new_tokens: t.Optional[int] = pydantic.Field(
        None, description="The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt."
    )
    early_stopping: bool = pydantic.Field(
        False,
        description="""Controls the stopping condition for beam-based methods, like beam-search. It accepts the 
        following values: 
        - `True`, where the generation stops as soon as there are `num_beams` complete candidates; 
        - `False`, where an heuristic is applied and the generation stops when is it very unlikely to find 
            better candidates; 
        - `"never"`, where the beam search procedure only stops when there cannot be better candidates 
            (canonical beam search algorithm)
    """,
    )
    max_time: t.Optional[float] = pydantic.Field(
        None,
        description="""The maximum amount of time you allow the computation to run for in seconds. generation will 
        still finish the current pass after allocated time has been passed.""",
    )

    # NOTE: Parameters for controling generaiton strategies
    num_beams: int = pydantic.Field(1, description="Number of beams for beam search. 1 means no beam search.")
    num_beam_groups: int = pydantic.Field(
        1,
        description="""Number of groups to divide `num_beams` into in order to ensure diversity among different 
        groups of beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.""",
    )
    penalty_alpha: t.Optional[float] = pydantic.Field(
        None,
        description="""The values balance the model confidence and the degeneration penalty in 
        contrastive search decoding.""",
    )
    use_cache: bool = pydantic.Field(
        True,
        description="""Whether or not the model should use the past last 
        key/values attentions (if applicable to the model) to speed up decoding.""",
    )

    # NOTE: Parameters for manipulation of the model output logits
    temperature: float = pydantic.Field(
        1.0, ge=0.0, le=1.0, description="The value used to modulate the next token probabilities."
    )
    top_k: int = pydantic.Field(
        50, description="The number of highest probability vocabulary tokens to keep for top-k-filtering."
    )
    top_p: float = pydantic.Field(
        1.0,
        description="""If set to float < 1, only the smallest set of most probable tokens with 
        probabilities that add up to `top_p` or higher are kept for generation.""",
    )
    typical_p: float = pydantic.Field(
        1.0,
        description="""Local typicality measures how similar the conditional probability of predicting a target 
        token next is to the expected conditional probability of predicting a random token next, given the 
        partial text already generated. If set to float < 1, the smallest set of the most locally typical 
        tokens with probabilities that add up to `typical_p` or higher are kept for generation. See [this
        paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
    """,
    )
    epsilon_cutoff: float = pydantic.Field(
        0.0,
        description="""\
        If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
        `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the
        size of the model. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) 
        for more details.
    """,
    )
    eta_cutoff: float = pydantic.Field(
        0.0,
        description="""Eta sampling is a hybrid of locally typical sampling and epsilon sampling. 
        If set to float strictly between 0 and 1, a token is only considered if it is greater than 
        either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`. The latter term is 
        intuitively the expected next token probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested 
        values range from 3e-4 to 2e-3, depending on the size of the model. 
        See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more details.
    """,
    )
    diversity_penalty: float = pydantic.Field(
        0.0,
        description="""This value is subtracted from a beam's score if it generates a token same 
        as any beam from other group at a particular time. Note that `diversity_penalty` is only 
        effective if `group beam search` is enabled.
    """,
    )
    repetition_penalty: float = pydantic.Field(
        1.0,
        description="""The parameter for repetition penalty. 1.0 means no penalty. 
        See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.""",
    )
    encoder_repetition_penalty: float = pydantic.Field(
        1.0,
        description="""The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are 
        not in the original input. 1.0 means no penalty.""",
    )
    length_penalty: float = pydantic.Field(
        1.0,
        description="""Exponential penalty to the length that is used with beam-based generation. It is applied 
        as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since 
        the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer 
        sequences, while `length_penalty` < 0.0 encourages shorter sequences.
    """,
    )
    no_repeat_ngram_size: int = pydantic.Field(
        0, description="If set to int > 0, all ngrams of that size can only occur once."
    )
    bad_words_ids: t.Optional[t.List[t.List[int]]] = pydantic.Field(
        None,
        description="""List of token ids that are not allowed to be generated. In order to get the token ids 
        of the words that should not appear in the generated text, use 
        `tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids`.
        """,
    )
    force_words_ids: t.Optional[t.Union[t.List[t.List[int]], t.List[t.List[t.List[int]]]]] = pydantic.Field(
        None,
        description="""List of token ids that must be generated. If given a `List[List[int]]`, this is treated 
        as a simple list of words that must be included, the opposite to `bad_words_ids`. 
        If given `List[List[List[int]]]`, this triggers a 
        [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one
        can allow different forms of each word.
        """,
    )
    renormalize_logits: bool = pydantic.Field(
        False,
        description="""Whether to renormalize the logits after applying all the logits processors or warpers 
        (including the custom ones). It's highly recommended to set this flag to `True` as the search 
        algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization.
    """,
    )
    constraints: t.Optional[t.List["Constraint"]] = pydantic.Field(
        None,
        description="""Custom constraints that can be added to the generation to ensure that the output 
        will contain the use of certain tokens as defined by ``Constraint`` objects, in the most sensible way possible.
        """,
    )
    forced_bos_token_id: t.Optional[int] = pydantic.Field(
        None,
        description="""The id of the token to force as the first generated token after the 
        ``decoder_start_token_id``. Useful for multilingual models like 
        [mBART](https://huggingface.co/docs/transformers/model_doc/mbart) where the first generated token needs 
        to be the target language token.
    """,
    )
    forced_eos_token_id: t.Optional[t.Union[int, t.List[int]]] = pydantic.Field(
        None,
        description="""The id of the token to force as the last generated token when `max_length` is reached. 
        Optionally, use a list to set multiple *end-of-sequence* tokens.""",
    )
    remove_invalid_values: bool = pydantic.Field(
        False,
        description="""Whether to remove possible *nan* and *inf* outputs of the model to prevent the 
        generation method to crash. Note that using `remove_invalid_values` can slow down generation.""",
    )
    exponential_decay_length_penalty: t.Optional[t.Tuple[int, float]] = pydantic.Field(
        None,
        description="""This tuple adds an exponentially increasing length penalty, after a certain amount of tokens 
        have been generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` 
        indicates where penalty starts and `decay_factor` represents the factor of exponential decay
    """,
    )
    suppress_tokens: t.Optional[t.List[int]] = pydantic.Field(
        None,
        description="""A list of tokens that will be suppressed at generation. The `SupressTokens` logit 
        processor will set their log probs to `-inf` so that they are not sampled.
    """,
    )
    begin_suppress_tokens: t.Optional[t.List[int]] = pydantic.Field(
        None,
        description="""A list of tokens that will be suppressed at the beginning of the generation. The 
        `SupressBeginTokens` logit processor will set their log probs to `-inf` so that they are not sampled.
        """,
    )
    forced_decoder_ids: t.Optional[t.List[t.List[int]]] = pydantic.Field(
        None,
        description="""A list of pairs of integers which indicates a mapping from generation indices to token indices 
        that will be forced before sampling. For example, `[[1, 123]]` means the second generated token will always 
        be a token of index 123.
        """,
    )

    # NOTE: Parameters that define the output variables of `generate`
    num_return_sequences: int = pydantic.Field(
        1, description="The number of independently computed returned sequences for each element in the batch."
    )
    output_attentions: bool = pydantic.Field(
        False,
        description="""Whether or not to return the attentions tensors of all attention layers. 
        See `attentions` under returned tensors for more details. """,
    )
    output_hidden_states: bool = pydantic.Field(
        False,
        description="""Whether or not to return the hidden states of all layers. 
        See `hidden_states` under returned tensors for more details.
        """,
    )
    output_scores: bool = pydantic.Field(
        False,
        description="""Whether or not to return the prediction scores. See `scores` under returned 
        tensors for more details.""",
    )

    # NOTE: Generation parameters exclusive to encoder-decoder models
    encoder_no_repeat_ngram_size: int = pydantic.Field(
        0,
        description="""If set to int > 0, all ngrams of that size that occur in the 
        `encoder_input_ids` cannot occur in the `decoder_input_ids`.
        """,
    )
    decoder_start_token_id: t.Optional[int] = pydantic.Field(
        None,
        description="""If an encoder-decoder model starts decoding with a 
        different token than *bos*, the id of that token.
        """,
    )

    # NOTE: pydantic definition
    model_config = dict(arbitrary_types_allowed=True, extra="forbid")

    if t.TYPE_CHECKING:
        # The following is handled via __pydantic_init_subclass__
        __openllm_env_name__: str
        __openllm_model_name__: str

    def __init_subclass__(cls, *, _internal: bool = False, **kwargs: t.Any) -> None:
        if not _internal:
            raise RuntimeError(
                "GenerationConfig is not meant to be used directly, "
                "but you can access this via a LLMConfig.generation_config"
            )
        model_name = kwargs.get("model_name", None)
        if model_name is None:
            raise RuntimeError("Failed to initialize GenerationConfig subclass (missing model_name)")
        cls.__openllm_model_name__ = inflection.underscore(model_name)
        cls.__openllm_env_name__ = cls.__openllm_model_name__.upper()

    @classmethod
    def construct_from_llm_config(cls, llm_config: type[LLMConfig]) -> GenerationConfig:
        """Parse ModelConfig.GenerationConfig into a GenerationConfig object."""
        return cls.model_validate(
            {
                k: v
                for k, v in vars(llm_config.GenerationConfig).items()
                if not k.startswith("_") and k in cls.model_fields
            }
        )

    def model_post_init(self, _: t.Any):
        # NOTE: I don't know how to do this more efficiently in pydantic v2 yet, will probably
        # need to consult the pydantic team on this.
        for key, field in self.model_fields.items():
            json_schema: dict[str, t.Any] = (
                copy.deepcopy(field.json_schema_extra) if field.json_schema_extra is not None else {}
            )
            env_key = f"OPENLLM_{self.__openllm_env_name__}_GENERATION_{key.upper()}"
            if "env" in json_schema:
                field.default = os.environ.get(json_schema["env"], field.default)
                continue
            json_schema["env"] = env_key
            # then assign json_schema back to field
            field.json_schema_extra = json_schema
            field.default = os.environ.get(env_key, field.default)

    def to_click_options(self, f: F[P]) -> t.Callable[[t.Callable[..., t.Any]], click.Command]:
        for name, field in self.model_fields.items():
            if t.get_origin(field.annotation) is t.Union:
                # NOTE: Union type is currently not yet supported, we probably just need to use environment instead.
                continue
            f = field_to_options(name, field, self.__openllm_model_name__, suffix_generation=True)(f)
        return optgroup.group(
            f"{self.__class__.__name__} generation options",
            help=f"[Auto-generated from '{self.__class__.__qualname__}']",
        )(f)


class LLMConfig(pydantic.BaseModel, ABC):
    # NOTE: we need this to allow such that we can parse generation_config from configuration.
    model_config = {"extra": "allow"}

    def __getattr__(self, attr: t.Any) -> t.Any:
        if attr in self.generation_config.model_fields:
            return getattr(self.generation_config, attr)
        return getattr(self, attr)

    def __repr_args__(self) -> ReprArgs:
        """Overwrite from default BaseModel and don't show __pydantic_extra__."""
        yield from (
            (k, v)
            for k, v in self.__dict__.items()
            if not k.startswith("_") and (k not in self.model_fields or self.model_fields[k].repr)
        )
        yield from (
            (k, v)
            for k, v in self.generation_config.__dict__.items()
            if not k.startswith("_")
            and (k not in self.generation_config.model_fields or self.generation_config.model_fields[k].repr)
        )
        yield from ((k, getattr(self, k)) for k, v in self.model_computed_fields.items() if v.repr)

    if t.TYPE_CHECKING:
        # The following is handled via __pydantic_init_subclass__, and is only used for TYPE_CHECKING
        __openllm_model_name__: str = ""
        __openllm_start_name__: str = ""
        __openllm_timeout__: int = 0
        GenerationConfig: type[t.Any] = GenerationConfig

    def __init_subclass__(cls, *, default_timeout: int | None = None, **kwargs: t.Any):
        if default_timeout is None:
            default_timeout = 3600
        cls.__openllm_timeout__ = default_timeout

        super(LLMConfig, cls).__init_subclass__(**kwargs)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: t.Any):
        cls.__openllm_model_name__ = inflection.underscore(cls.__name__.replace("Config", ""))
        cls.__openllm_start_name__ = inflection.dasherize(cls.__openllm_model_name__)
        if hasattr(cls, "GenerationConfig"):
            generation_class = t.cast(
                "type[GenerationConfig]",
                types.new_class(
                    cls.__name__.replace("Config", "") + "GenerationConfig",
                    (GenerationConfig,),
                    {"model_name": cls.__openllm_model_name__, "_internal": True},
                ),
            )
            cls.generation_config = generation_class.construct_from_llm_config(cls)
            delattr(cls, "GenerationConfig")

        for key, field in cls.model_fields.items():
            if not field.json_schema_extra:
                field.json_schema_extra = {}
            if "env" in field.json_schema_extra:
                continue
            field.json_schema_extra["env"] = f"OPENLLM_{cls.__openllm_model_name__.upper()}_{key.upper()}"

    def model_post_init(self, _: t.Any):
        if self.__pydantic_extra__:
            generation_config = self.__pydantic_extra__.pop("generation_config", None)
            if generation_config is not None:
                assert isinstance(generation_config, dict), "generation_config must be a dict."
                self.generation_config = self.generation_config.model_copy(
                    update=t.cast("dict[str, t.Any]", generation_config), deep=True
                )
            else:
                # The rest of the extras fields should just be the generation_config.
                self.generation_config = self.generation_config.model_copy(update=self.__pydantic_extra__, deep=True)
                # NOTE: Non need to maintain key from __pydantic_extra__ that is already parsed into generation_config
                _object_setattr(
                    self,
                    "__pydantic_extra__",
                    {k: v for k, v in self.__pydantic_extra__.items() if k not in self.generation_config.model_fields},
                )

    def model_dump_yaml(self):
        try:
            return yaml.safe_dump(self.model_dump(), sort_keys=False)
        except yaml.YAMLError as e:
            raise openllm.exceptions.ValidationError(f"Failed to dump configuration to yaml: {e}") from e

    @classmethod
    def model_validate_yaml(cls, path: str, ctx: str | None = None, strict: bool = True) -> LLMConfig:
        with open(openllm.utils.resolve_user_filepath(path, ctx=ctx), "rb") as f:
            content = yaml.safe_load(f)
            try:
                return cls.model_validate(content, strict=strict)
            except pydantic.ValidationError as e:
                raise openllm.exceptions.ValidationError(f"Failed to parse configuration to {cls}: {e}") from e

    def model_dump(self, **kwargs: t.Any):
        try:
            to_dump = super().model_dump(**kwargs)
            generation_config = self.generation_config.model_dump(exclude_defaults=True)
            to_dump["generation_config"] = generation_config
            return to_dump
        except pydantic.ValidationError as e:
            raise openllm.exceptions.ValidationError(f"Failed to dump configuration to dict: {e}") from e

    def with_options(self, __llm_config__: LLMConfig | None = None, **kwargs: t.Any) -> LLMConfig:
        """A helpers that respect configuration values that
        sets from environment variables for any given configuration class.
        """
        from_env_ = self.from_env()
        # filtered out None values
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if __llm_config__ is not None:
            kwargs = {**__llm_config__.model_dump(), **kwargs}
        if from_env_:
            return from_env_.model_construct(**kwargs)
        return self.model_construct(**kwargs)

    @classmethod
    def from_env(cls) -> LLMConfig | None:
        envvar = openllm.utils.MODEL_CONFIG_ENV_VAR(cls.__openllm_model_name__)
        env_json_string = os.environ.get(envvar, None)
        if env_json_string is None:
            return

        try:
            return cls.model_construct(**orjson.loads(env_json_string))
        except pydantic.ValidationError as e:
            raise RuntimeError(f"Failed to parse environment variable '{envvar}' as a valid JSON string.") from e

    def model_validate_click(self, **attrs: t.Any) -> tuple[LLMConfig, dict[str, t.Any]]:
        """Parse given click attributes into a LLMConfig and return the remaining click attributes."""
        llm_config_kwargs = {
            k[len(self.__openllm_model_name__) + 1 :]: v
            for k, v in attrs.items()
            if k[len(self.__openllm_model_name__) + 1 :] in self.model_fields
        }
        llm_config_kwargs["generation_config"] = {
            k[len(self.__openllm_model_name__ + "_generation") + 1 :]: v
            for k, v in attrs.items()
            if k[len(self.__openllm_model_name__ + "_generation") + 1 :] in self.generation_config.model_fields
        }
        return self.with_options(**llm_config_kwargs), {
            k: v for k, v in attrs.items() if not k.startswith(self.__openllm_model_name__)
        }

    def to_generation_config(self) -> transformers.GenerationConfig:
        return transformers.GenerationConfig(**self.generation_config.model_dump())

    def to_click_options(self, f: F[P]) -> t.Callable[[t.Callable[..., t.Any]], click.Command]:
        """
        Convert current model to click options. This can be used as a decorator for click commands.
        Note that the identifier for all LLMConfig will be prefixed with '<model_name>_*', and the generation config
        will be prefixed with '<model_name>_generation_*'.
        """
        wrapped_generation = self.generation_config.to_click_options(f)
        if len(self.model_fields.values()) == 0:
            return wrapped_generation
        for name, field in self.model_fields.items():
            wrapped_generation = field_to_options(name, field, self.__openllm_model_name__)(wrapped_generation)
        return optgroup.group(
            f"{self.__class__.__name__} options", help=f"[Auto-generated from '{self.__class__.__qualname__}']"
        )(wrapped_generation)
