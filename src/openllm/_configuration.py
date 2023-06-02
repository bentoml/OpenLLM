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

    class GenerationConfig:
        temperature: float = 0.75
        max_new_tokens: int = 3000
        top_k: int = 50
        top_p: float = 0.4
        repetition_penalty = 1.0
```


"""
from __future__ import annotations

import logging
import os
import typing as t
from operator import itemgetter

import attr
import click
import inflection
import orjson
from bentoml._internal.models.model import ModelSignature
from cattr.gen import make_dict_unstructure_fn, override
from click_option_group import optgroup

import openllm

from .exceptions import GpuNotAvailableError, OpenLLMException
from .utils import LazyType, ModelEnv, bentoml_cattr, dantic, lenient_issubclass

if t.TYPE_CHECKING:
    import tensorflow as tf
    import torch
    import transformers
    from attr import _CountingAttr, _make_init
    from transformers.generation.beam_constraints import Constraint

    P = t.ParamSpec("P")

    F = t.Callable[P, t.Any]

    ReprArgs: t.TypeAlias = t.Iterable[tuple[str | None, t.Any]]

    DictStrAny = dict[str, t.Any]
    ItemgetterAny = itemgetter[t.Any]
else:
    Constraint = t.Any
    DictStrAny = dict
    ItemgetterAny = itemgetter
    # NOTE: Using internal API from attr here, since we are actually
    # allowing subclass of openllm.LLMConfig to become 'attrs'-ish
    from attr._make import _CountingAttr, _make_init

    transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")
    torch = openllm.utils.LazyLoader("torch", globals(), "torch")
    tf = openllm.utils.LazyLoader("tf", globals(), "tensorflow")

__all__ = ["LLMConfig", "ModelSignature"]

_object_setattr = object.__setattr__

logger = logging.getLogger(__name__)


def attrs_to_options(
    name: str,
    field: attr.Attribute[t.Any],
    model_name: str,
    suffix_generation: bool = False,
) -> t.Callable[[F[P]], F[P]]:
    # TODO: support parsing nested attrs class
    envvar = field.metadata["env"]
    dasherized = inflection.dasherize(name)
    underscored = inflection.underscore(name)

    full_option_name = f"--{dasherized}"
    if field.type is bool:
        full_option_name += f"/--no-{dasherized}"
    if suffix_generation:
        identifier = f"{model_name}_generation_{underscored}"
    else:
        identifier = f"{model_name}_{underscored}"

    return optgroup.option(
        identifier,
        full_option_name,
        type=field.type,
        required=field.default is attr.NOTHING,
        default=field.default if field.default not in (attr.NOTHING, None) else None,
        show_default=True,
        multiple=dantic.allows_multiple(field.type),
        help=field.metadata.get("description", "(No description provided)"),
        show_envvar=True,
        envvar=envvar,
    )


@attr.define
class GenerationConfig:
    """Generation config provides the configuration to then be parsed to ``transformers.GenerationConfig``,
    with some additional validation and environment constructor.

    Note that we always set `do_sample=True`
    """

    # NOTE: parameters for controlling the length of the output
    max_new_tokens: int = dantic.Field(
        20,
        ge=0,
        description="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.",
    )
    min_length: int = dantic.Field(
        0,
        ge=0,
        description="""The minimum length of the sequence to be generated. Corresponds to the length of the 
        input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.""",
    )
    min_new_tokens: int = dantic.Field(
        description="The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.",
    )
    early_stopping: bool = dantic.Field(
        False,
        description="""Controls the stopping condition for beam-based methods, like beam-search. It accepts the 
        following values: `True`, where the generation stops as soon as there are `num_beams` complete candidates; 
        `False`, where an heuristic is applied and the generation stops when is it very unlikely to find 
        better candidates; `"never"`, where the beam search procedure only stops when there 
        cannot be better candidates (canonical beam search algorithm)
    """,
    )
    max_time: float = dantic.Field(
        description="""The maximum amount of time you allow the computation to run for in seconds. generation will 
        still finish the current pass after allocated time has been passed.""",
    )

    # NOTE: Parameters for controling generaiton strategies
    num_beams: int = dantic.Field(1, description="Number of beams for beam search. 1 means no beam search.")
    num_beam_groups: int = dantic.Field(
        1,
        description="""Number of groups to divide `num_beams` into in order to ensure diversity among different 
        groups of beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details.""",
    )
    penalty_alpha: float = dantic.Field(
        description="""The values balance the model confidence and the degeneration penalty in 
        contrastive search decoding.""",
    )
    use_cache: bool = dantic.Field(
        True,
        description="""Whether or not the model should use the past last 
        key/values attentions (if applicable to the model) to speed up decoding.""",
    )

    # NOTE: Parameters for manipulation of the model output logits
    temperature: float = dantic.Field(
        1.0, ge=0.0, le=1.0, description="The value used to modulate the next token probabilities."
    )
    top_k: int = dantic.Field(
        50, description="The number of highest probability vocabulary tokens to keep for top-k-filtering."
    )
    top_p: float = dantic.Field(
        1.0,
        description="""If set to float < 1, only the smallest set of most probable tokens with 
        probabilities that add up to `top_p` or higher are kept for generation.""",
    )
    typical_p: float = dantic.Field(
        1.0,
        description="""Local typicality measures how similar the conditional probability of predicting a target 
        token next is to the expected conditional probability of predicting a random token next, given the 
        partial text already generated. If set to float < 1, the smallest set of the most locally typical 
        tokens with probabilities that add up to `typical_p` or higher are kept for generation. See [this
        paper](https://arxiv.org/pdf/2202.00666.pdf) for more details.
    """,
    )
    epsilon_cutoff: float = dantic.Field(
        0.0,
        description="""\
        If set to float strictly between 0 and 1, only tokens with a conditional probability greater than
        `epsilon_cutoff` will be sampled. In the paper, suggested values range from 3e-4 to 9e-4, depending on the
        size of the model. See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) 
        for more details.
    """,
    )
    eta_cutoff: float = dantic.Field(
        0.0,
        description="""Eta sampling is a hybrid of locally typical sampling and epsilon sampling. 
        If set to float strictly between 0 and 1, a token is only considered if it is greater than 
        either `eta_cutoff` or `sqrt(eta_cutoff) * exp(-entropy(softmax(next_token_logits)))`. The latter term is 
        intuitively the expected next token probability, scaled by `sqrt(eta_cutoff)`. In the paper, suggested 
        values range from 3e-4 to 2e-3, depending on the size of the model. 
        See [Truncation Sampling as Language Model Desmoothing](https://arxiv.org/abs/2210.15191) for more details.
    """,
    )
    diversity_penalty: float = dantic.Field(
        0.0,
        description="""This value is subtracted from a beam's score if it generates a token same 
        as any beam from other group at a particular time. Note that `diversity_penalty` is only 
        effective if `group beam search` is enabled.
    """,
    )
    repetition_penalty: float = dantic.Field(
        1.0,
        description="""The parameter for repetition penalty. 1.0 means no penalty. 
        See [this paper](https://arxiv.org/pdf/1909.05858.pdf) for more details.""",
    )
    encoder_repetition_penalty: float = dantic.Field(
        1.0,
        description="""The paramater for encoder_repetition_penalty. An exponential penalty on sequences that are 
        not in the original input. 1.0 means no penalty.""",
    )
    length_penalty: float = dantic.Field(
        1.0,
        description="""Exponential penalty to the length that is used with beam-based generation. It is applied 
        as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since 
        the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer 
        sequences, while `length_penalty` < 0.0 encourages shorter sequences.
    """,
    )
    no_repeat_ngram_size: int = dantic.Field(
        0, description="If set to int > 0, all ngrams of that size can only occur once."
    )
    bad_words_ids: t.List[t.List[int]] = dantic.Field(
        description="""List of token ids that are not allowed to be generated. In order to get the token ids 
        of the words that should not appear in the generated text, use 
        `tokenizer(bad_words, add_prefix_space=True, add_special_tokens=False).input_ids`.
        """,
    )

    # NOTE: t.Union is not yet supported on CLI, but the environment variable should already be available.
    force_words_ids: t.Union[t.List[t.List[int]], t.List[t.List[t.List[int]]]] = dantic.Field(
        description="""List of token ids that must be generated. If given a `List[List[int]]`, this is treated 
        as a simple list of words that must be included, the opposite to `bad_words_ids`. 
        If given `List[List[List[int]]]`, this triggers a 
        [disjunctive constraint](https://github.com/huggingface/transformers/issues/14081), where one
        can allow different forms of each word.
        """,
    )
    renormalize_logits: bool = dantic.Field(
        False,
        description="""Whether to renormalize the logits after applying all the logits processors or warpers 
        (including the custom ones). It's highly recommended to set this flag to `True` as the search 
        algorithms suppose the score logits are normalized but some logit processors or warpers break the normalization.
    """,
    )
    constraints: t.List[Constraint] = dantic.Field(
        description="""Custom constraints that can be added to the generation to ensure that the output
        will contain the use of certain tokens as defined by ``Constraint`` objects, in the most sensible way possible.
        """,
    )
    forced_bos_token_id: int = dantic.Field(
        description="""The id of the token to force as the first generated token after the 
        ``decoder_start_token_id``. Useful for multilingual models like 
        [mBART](https://huggingface.co/docs/transformers/model_doc/mbart) where the first generated token needs 
        to be the target language token.
    """,
    )
    forced_eos_token_id: t.Union[int, t.List[int]] = dantic.Field(
        description="""The id of the token to force as the last generated token when `max_length` is reached. 
        Optionally, use a list to set multiple *end-of-sequence* tokens.""",
    )
    remove_invalid_values: bool = dantic.Field(
        False,
        description="""Whether to remove possible *nan* and *inf* outputs of the model to prevent the 
        generation method to crash. Note that using `remove_invalid_values` can slow down generation.""",
    )
    exponential_decay_length_penalty: t.Tuple[int, float] = dantic.Field(
        description="""This tuple adds an exponentially increasing length penalty, after a certain amount of tokens 
        have been generated. The tuple shall consist of: `(start_index, decay_factor)` where `start_index` 
        indicates where penalty starts and `decay_factor` represents the factor of exponential decay
    """,
    )
    suppress_tokens: t.List[int] = dantic.Field(
        description="""A list of tokens that will be suppressed at generation. The `SupressTokens` logit 
        processor will set their log probs to `-inf` so that they are not sampled.
    """,
    )
    begin_suppress_tokens: t.List[int] = dantic.Field(
        description="""A list of tokens that will be suppressed at the beginning of the generation. The 
        `SupressBeginTokens` logit processor will set their log probs to `-inf` so that they are not sampled.
        """,
    )
    forced_decoder_ids: t.List[t.List[int]] = dantic.Field(
        description="""A list of pairs of integers which indicates a mapping from generation indices to token indices 
        that will be forced before sampling. For example, `[[1, 123]]` means the second generated token will always 
        be a token of index 123.
        """,
    )

    # NOTE: Parameters that define the output variables of `generate`
    num_return_sequences: int = dantic.Field(
        1, description="The number of independently computed returned sequences for each element in the batch."
    )
    output_attentions: bool = dantic.Field(
        False,
        description="""Whether or not to return the attentions tensors of all attention layers. 
        See `attentions` under returned tensors for more details. """,
    )
    output_hidden_states: bool = dantic.Field(
        False,
        description="""Whether or not to return the hidden states of all layers. 
        See `hidden_states` under returned tensors for more details.
        """,
    )
    output_scores: bool = dantic.Field(
        False,
        description="""Whether or not to return the prediction scores. See `scores` under returned 
        tensors for more details.""",
    )

    # NOTE: Special tokens that can be used at generation time
    pad_token_id: int = dantic.Field(description="The id of the *padding* token.")
    bos_token_id: int = dantic.Field(description="The id of the *beginning-of-sequence* token.")
    eos_token_id: t.Union[int, t.List[int]] = dantic.Field(
        description="""The id of the *end-of-sequence* token. Optionally, use a list to set 
        multiple *end-of-sequence* tokens.""",
    )

    # NOTE: Generation parameters exclusive to encoder-decoder models
    encoder_no_repeat_ngram_size: int = dantic.Field(
        0,
        description="""If set to int > 0, all ngrams of that size that occur in the 
        `encoder_input_ids` cannot occur in the `decoder_input_ids`.
        """,
    )
    decoder_start_token_id: int = dantic.Field(
        description="""If an encoder-decoder model starts decoding with a 
        different token than *bos*, the id of that token.
        """,
    )

    if t.TYPE_CHECKING:

        def __attrs_init__(self, **_: t.Any):
            ...

    def __init__(self, *, _internal: bool = False, **attrs: t.Any):
        if not _internal:
            raise RuntimeError(
                "GenerationConfig is not meant to be used directly, "
                "but you can access this via a LLMConfig.generation_config"
            )
        self.__attrs_init__(**attrs)


bentoml_cattr.register_unstructure_hook_factory(
    lambda cls: attr.has(cls) and lenient_issubclass(cls, GenerationConfig),
    lambda cls: make_dict_unstructure_fn(
        cls,
        bentoml_cattr,
        **{k: override(omit=True) for k, v in attr.fields_dict(cls).items() if v.default in (None, attr.NOTHING)},
    ),
)


def _populate_value_from_env_var(
    key: str, transform: t.Callable[[str], str] | None = None, fallback: t.Any = None
) -> t.Any:
    if transform is not None and callable(transform):
        key = transform(key)

    return os.environ.get(key, fallback)


def env_transformers(cls: type[GenerationConfig], fields: list[attr.Attribute[t.Any]]) -> list[attr.Attribute[t.Any]]:
    transformed: list[attr.Attribute[t.Any]] = []
    for f in fields:
        if "env" not in f.metadata:
            raise ValueError(
                "Make sure to setup the field with 'cls.Field' or 'attr.field(..., metadata={\"env\": \"...\"})'"
            )
        _from_env = _populate_value_from_env_var(f.metadata["env"])
        if _from_env is not None:
            f = f.evolve(default=_from_env)
        transformed.append(f)
    return transformed


# sentinel object for unequivocal object() getattr
_sentinel = object()


def _has_own_attribute(cls: type[t.Any], attrib_name: t.Any):
    """
    Check whether *cls* defines *attrib_name* (and doesn't just inherit it).
    """
    attr = getattr(cls, attrib_name, _sentinel)
    if attr is _sentinel:
        return False

    for base_cls in cls.__mro__[1:]:
        a = getattr(base_cls, attrib_name, None)
        if attr is a:
            return False

    return True


def _get_annotations(cls: type[t.Any]) -> DictStrAny:
    """
    Get annotations for *cls*.
    """
    if _has_own_attribute(cls, "__annotations__"):
        return cls.__annotations__

    return DictStrAny()


# The below is vendorred from attrs
def _collect_base_attrs(
    cls: type[LLMConfig], taken_attr_names: set[str]
) -> tuple[list[attr.Attribute[t.Any]], dict[str, type[t.Any]]]:
    """
    Collect attr.ibs from base classes of *cls*, except *taken_attr_names*.
    """
    base_attrs: list[attr.Attribute[t.Any]] = []
    base_attr_map: dict[str, type[t.Any]] = {}  # A dictionary of base attrs to their classes.

    # Traverse the MRO and collect attributes.
    for base_cls in reversed(cls.__mro__[1:-1]):
        for a in getattr(base_cls, "__attrs_attrs__", []):
            if a.inherited or a.name in taken_attr_names:
                continue

            a = a.evolve(inherited=True)
            base_attrs.append(a)
            base_attr_map[a.name] = base_cls

    # For each name, only keep the freshest definition i.e. the furthest at the back.
    filtered: list[attr.Attribute[t.Any]] = []
    seen: set[str] = set()
    for a in reversed(base_attrs):
        if a.name in seen:
            continue
        filtered.insert(0, a)
        seen.add(a.name)

    return filtered, base_attr_map


_classvar_prefixes = (
    "typing.ClassVar",
    "t.ClassVar",
    "ClassVar",
    "typing_extensions.ClassVar",
)


def _is_class_var(annot: str | t.Any) -> bool:
    """
    Check whether *annot* is a typing.ClassVar.

    The string comparison hack is used to avoid evaluating all string
    annotations which would put attrs-based classes at a performance
    disadvantage compared to plain old classes.
    """
    annot = str(annot)

    # Annotation can be quoted.
    if annot.startswith(("'", '"')) and annot.endswith(("'", '"')):
        annot = annot[1:-1]

    return annot.startswith(_classvar_prefixes)


def _add_method_dunders(cls: type[t.Any], method: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
    """
    Add __module__ and __qualname__ to a *method* if possible.
    """
    try:
        method.__module__ = cls.__module__
    except AttributeError:
        pass

    try:
        method.__qualname__ = ".".join((cls.__qualname__, method.__name__))
    except AttributeError:
        pass

    try:
        method.__doc__ = "Method generated by attrs for class " f"{cls.__qualname__}."
    except AttributeError:
        pass

    return method


# NOTE: vendorred from attrs
def _compile_and_eval(script: str, globs: dict[str, t.Any], locs: dict[str, t.Any] | None = None, filename: str = ""):
    """
    "Exec" the script with the given global (globs) and local (locs) variables.
    """
    bytecode = compile(script, filename, "exec")
    eval(bytecode, globs, locs)


def _make_attr_tuple_class(cls_name: str, attr_names: t.Iterable[str]) -> type[tuple[attr.Attribute[t.Any], ...]]:
    """
    Create a tuple subclass to hold `Attribute`s for an `attrs` class.

    The subclass is a bare tuple with properties for names.

    class MyClassAttributes(tuple):
        __slots__ = ()
        x = property(itemgetter(0))
    """
    attr_class_name = f"{cls_name}Attributes"
    attr_class_template = [
        f"class {attr_class_name}(tuple):",
        "    __slots__ = ()",
    ]
    if attr_names:
        for i, attr_name in enumerate(attr_names):
            attr_class_template.append(f"    {attr_name} = _attrs_property(_attrs_itemgetter({i}))")
    else:
        attr_class_template.append("    pass")
    globs: dict[str, t.Any] = {"_attrs_itemgetter": ItemgetterAny, "_attrs_property": property}
    _compile_and_eval("\n".join(attr_class_template), globs)
    return globs[attr_class_name]


def _make_internal_generation_class(cls: type[LLMConfig]) -> type[GenerationConfig]:
    _has_gen_class = _has_own_attribute(cls, "GenerationConfig")

    hints = t.get_type_hints(GenerationConfig)

    def _evolve_with_base_default(
        __cls: type[GenerationConfig], fields: list[attr.Attribute[t.Any]]
    ) -> list[attr.Attribute[t.Any]]:
        transformed: list[attr.Attribute[t.Any]] = []
        for f in fields:
            env = f"OPENLLM_{cls.__openllm_model_name__.upper()}_GENERATION_{f.name.upper()}"
            _from_env = _populate_value_from_env_var(env, fallback=f.default)
            default_value = f.default if not _has_gen_class else getattr(cls.GenerationConfig, f.name, _from_env)
            transformed.append(f.evolve(default=default_value, metadata={"env": env}, type=hints[f.name]))
        return transformed

    generated_cls = attr.make_class(
        cls.__name__.replace("Config", "GenerationConfig"),
        [],
        bases=(GenerationConfig,),
        frozen=True,
        slots=True,
        repr=True,
        field_transformer=_evolve_with_base_default,
    )

    return generated_cls


@attr.define
class LLMConfig:
    Field = dantic.Field
    """Field is a alias to the internal dantic utilities to easily create
    attrs.fields with pydantic-compatible interface.
    """

    if t.TYPE_CHECKING:

        def __attrs_init__(self, **attrs: t.Any):
            ...

        # The following is handled via __init_subclass__, and is only used for TYPE_CHECKING
        __attrs_attrs__: tuple[attr.Attribute[t.Any], ...] = tuple()
        __openllm_attrs__: tuple[str, ...] = tuple()

        __openllm_timeout__: int = 3600
        __openllm_requires_gpu__: bool = False
        __openllm_trust_remote_code__: bool = False

        __openllm_model_name__: str = ""
        __openllm_start_name__: str = ""
        __openllm_name_type__: t.Literal["dasherize", "lowercase"] = "dasherize"

        __openllm_env__: openllm.utils.ModelEnv = Field(None, init=False)

        generation_class: type[GenerationConfig] = Field(None, init=False)

        GenerationConfig: type = type

    def __init_subclass__(
        cls,
        *,
        name_type: t.Literal["dasherize", "lowercase"] = "dasherize",
        default_timeout: int | None = None,
        trust_remote_code: bool = False,
        requires_gpu: bool = False,
    ):
        if name_type == "dasherize":
            model_name = inflection.underscore(cls.__name__.replace("Config", ""))
            start_name = inflection.dasherize(model_name)
        else:
            model_name = cls.__name__.replace("Config", "").lower()
            start_name = model_name

        cls.__openllm_name_type__ = name_type
        cls.__openllm_requires_gpu__ = requires_gpu
        cls.__openllm_timeout__ = default_timeout or 3600
        cls.__openllm_trust_remote_code__ = trust_remote_code

        cls.__openllm_model_name__ = model_name
        cls.__openllm_start_name__ = start_name
        cls.__openllm_env__ = openllm.utils.ModelEnv(model_name)

        # NOTE: Since we want to enable a pydantic-like experience
        # this means we will have to hide the attr abstraction, and generate
        # all of the Field from __init_subclass__
        # Some of the logics here are from attr._make._transform_attrs
        anns = _get_annotations(cls)
        cd = cls.__dict__

        def field_env_key(key: str) -> str:
            return f"OPENLLM_{model_name.upper()}_{key.upper()}"

        ca_names = {name for name, attr in cd.items() if isinstance(attr, _CountingAttr)}
        ca_list: list[tuple[str, _CountingAttr[t.Any]]] = []
        annotated_names: set[str] = set()
        for attr_name, typ in anns.items():
            if _is_class_var(typ):
                continue
            annotated_names.add(attr_name)
            val = cd.get(attr_name, attr.NOTHING)
            if not LazyType["_CountingAttr[t.Any]"](_CountingAttr).isinstance(val):
                if val is attr.NOTHING:
                    val = cls.Field(env=field_env_key(attr_name))
                else:
                    val = cls.Field(default=val, env=field_env_key(attr_name))
            ca_list.append((attr_name, val))
        unannotated = ca_names - annotated_names

        if len(unannotated) > 0:
            missing_annotated = sorted(unannotated, key=lambda n: t.cast("_CountingAttr[t.Any]", cd.get(n)).counter)
            raise openllm.exceptions.MissingAnnotationAttributeError(
                f"The following field doesn't have a type annotation: {missing_annotated}"
            )

        hints = t.get_type_hints(cls)
        # NOTE: we know need to determine the list of the attrs
        # by mro to at the very least support inheritance. Tho it is not recommended.
        own_attrs: list[attr.Attribute[t.Any]] = []
        for attr_name, ca in ca_list:
            gen_attribute = attr.Attribute.from_counting_attr(name=attr_name, ca=ca, type=hints.get(attr_name))
            if attr_name in ca_names:
                metadata = ca.metadata
                metadata["env"] = field_env_key(attr_name)
                gen_attribute = gen_attribute.evolve(metadata=metadata)
            own_attrs.append(gen_attribute)

        base_attrs, base_attr_map = _collect_base_attrs(cls, {a.name for a in own_attrs})
        attrs: list[attr.Attribute[t.Any]] = own_attrs + base_attrs

        # Mandatory vs non-mandatory attr order only matters when they are part of
        # the __init__ signature and when they aren't kw_only (which are moved to
        # the end and can be mandatory or non-mandatory in any order, as they will
        # be specified as keyword args anyway). Check the order of those attrs:
        had_default = False
        for a in (a for a in attrs if a.init is not False and a.kw_only is False):
            if had_default is True and a.default is attr.NOTHING:
                raise ValueError(
                    "No mandatory attributes allowed after an attribute with a "
                    f"default value or factory.  Attribute in question: {a!r}"
                )

            if had_default is False and a.default is not attr.NOTHING:
                had_default = True

        # NOTE: Resolve the alias and default value from environment variable
        attrs = [
            a.evolve(
                alias=a.name.lstrip("_") if not a.alias else None,
                # NOTE: This is where we actually populate with the environment variable set for this attrs.
                default=_populate_value_from_env_var(a.name, transform=field_env_key, fallback=a.default),
            )
            for a in attrs
        ]

        _has_pre_init = bool(getattr(cls, "__attrs_pre_init__", False))
        _has_post_init = bool(getattr(cls, "__attrs_post_init__", False))

        # __openllm_attrs__ is a tracking tuple[attr.Attribute[t.Any]]
        # that we construct ourself.
        cls.__openllm_attrs__ = tuple(a.name for a in attrs)
        AttrsTuple = _make_attr_tuple_class(cls.__name__, cls.__openllm_attrs__)
        # NOTE: generate a __attrs_init__ for the subclass
        cls.__attrs_init__ = _add_method_dunders(
            cls,
            _make_init(
                cls,
                AttrsTuple(attrs),
                _has_pre_init,
                _has_post_init,
                False,
                True,
                True,
                base_attr_map,
                False,
                None,
                attrs_init=True,
            ),
        )
        cls.__attrs_attrs__ = AttrsTuple(attrs)

        # NOTE: Finally, set the generation_class for this given config.
        cls.generation_class = _make_internal_generation_class(cls)

    @property
    def name_type(self) -> t.Literal["dasherize", "lowercase"]:
        return self.__openllm_name_type__

    def __init__(
        self,
        *,
        generation_config: dict[str, t.Any] | None = None,
        __openllm_extras__: dict[str, t.Any] | None = None,
        **attrs: t.Any,
    ):
        to_exclude = list(attr.fields_dict(self.generation_class)) + list(self.__openllm_attrs__)
        self.__openllm_extras__ = __openllm_extras__ or {k: v for k, v in attrs.items() if k not in to_exclude}

        attrs = {k: v for k, v in attrs.items() if k not in self.__openllm_extras__ and v is not None}

        if generation_config is None:
            generation_config = {k: v for k, v in attrs.items() if k in attr.fields_dict(self.generation_class)}

        self.generation_config = self.generation_class(**generation_config)

        attrs = {k: v for k, v in attrs.items() if k not in generation_config}

        extras = set(attrs).difference(set(attr.fields_dict(self.__class__)))

        self.__attrs_init__(**{k: v for k, v in attrs.items() if k not in extras})

    def __repr__(self) -> str:
        bases = f"{self.__class__.__qualname__.rsplit('>.', 1)[-1]}(generation_config={repr(self.generation_class())}"
        if len(self.__openllm_attrs__) > 0:
            bases += ", " + ", ".join([f"{k}={getattr(self, k)}" for k in self.__openllm_attrs__]) + ")"
        else:
            bases += ")"
        return bases

    @classmethod
    def check_if_gpu_is_available(cls, implementation: t.Literal["pt", "tf", "flax"] = "pt", force: bool = False):
        try:
            if cls.__openllm_requires_gpu__ or force:
                if implementation in ("tf", "flax") and len(tf.config.list_physical_devices("GPU")) == 0:
                    raise OpenLLMException("Required GPU for given model")
                else:
                    if not torch.cuda.is_available():
                        raise OpenLLMException("Required GPU for given model")
            else:
                logger.debug(
                    f"{cls} doesn't requires GPU by default. If you still want to check for GPU, set 'force=True'"
                )
        except OpenLLMException:
            if force:
                msg = "GPU is not available"
            else:
                msg = f"{cls} only supports running with GPU (None available)."
            raise GpuNotAvailableError(msg) from None

    def model_dump(self, flatten: bool = False, **_: t.Any):
        dumped = bentoml_cattr.unstructure(self)
        generation_config = bentoml_cattr.unstructure(self.generation_config)
        if not flatten:
            dumped["generation_config"] = generation_config
        else:
            dumped.update(generation_config)
        return dumped

    def model_dump_json(self, **kwargs: t.Any):
        return orjson.dumps(self.model_dump(**kwargs))

    @classmethod
    def model_construct_env(cls, __llm_config__: LLMConfig | None = None, **attrs: t.Any) -> LLMConfig:
        """A helpers that respect configuration values that
        sets from environment variables for any given configuration class.
        """
        # NOTE: filter out None values
        attrs = {k: v for k, v in attrs.items() if v is not None}

        env = ModelEnv(cls.__openllm_model_name__)

        env_json_string = os.environ.get(env.model_config, None)

        if env_json_string is not None:
            try:
                config_from_env = orjson.loads(env_json_string)
            except orjson.JSONDecodeError as e:
                raise RuntimeError(f"Failed to parse '{env.model_config}' as valid JSON string.") from e
            config_from_env.update(attrs)
            return bentoml_cattr.structure(config_from_env, cls)

        if __llm_config__ is not None:
            # NOTE: We only hit this branch on server-side, to ensure per-request configuration
            # is respected.
            attrs.update(__llm_config__.model_dump(flatten=True))

        return bentoml_cattr.structure(attrs, cls)

    def model_validate_click(self, **attrs: t.Any) -> tuple[LLMConfig, dict[str, t.Any]]:
        """Parse given click attributes into a LLMConfig and return the remaining click attributes."""
        llm_config_attrs: dict[str, t.Any] = {"generation_config": {}}
        key_to_remove: list[str] = []

        for k, v in attrs.items():
            if k.startswith(f"{self.__openllm_model_name__}_generation_"):
                llm_config_attrs["generation_config"][k[len(self.__openllm_model_name__ + "_generation_") :]] = v
                key_to_remove.append(k)
            elif k.startswith(f"{self.__openllm_model_name__}_"):
                llm_config_attrs[k[len(self.__openllm_model_name__ + "_") :]] = v
                key_to_remove.append(k)

        return self.model_construct_env(**llm_config_attrs), {k: v for k, v in attrs.items() if k not in key_to_remove}

    @t.overload
    def to_generation_config(self, return_as_dict: t.Literal[True] = ...) -> dict[str, t.Any]:
        ...

    @t.overload
    def to_generation_config(self, return_as_dict: t.Literal[False] = ...) -> transformers.GenerationConfig:
        ...

    def to_generation_config(self, return_as_dict: bool = False) -> transformers.GenerationConfig | dict[str, t.Any]:
        config = transformers.GenerationConfig(**self.generation_config.model_dump())
        return config.to_dict() if return_as_dict else config

    def to_click_options(self, f: F[P]) -> t.Callable[[F[P]], click.Command]:
        """
        Convert current model to click options. This can be used as a decorator for click commands.
        Note that the identifier for all LLMConfig will be prefixed with '<model_name>_*', and the generation config
        will be prefixed with '<model_name>_generation_*'.
        """
        for name, field in attr.fields_dict(self.generation_class).items():
            if t.get_origin(field.type) is t.Union:
                # NOTE: Union type is currently not yet supported, we probably just need to use environment instead.
                continue
            f = attrs_to_options(name, field, self.__openllm_model_name__, suffix_generation=True)(f)
        f = optgroup.group(f"{self.generation_class.__name__} generation options")(f)

        if len(self.__class__.__openllm_attrs__) == 0:
            return f
        for name, field in attr.fields_dict(self.__class__).items():
            if t.get_origin(field.type) is t.Union:
                # NOTE: Union type is currently not yet supported, we probably just need to use environment instead.
                continue
            f = attrs_to_options(name, field, self.__openllm_model_name__)(f)

        return optgroup.group(f"{self.__class__.__name__} options")(f)


bentoml_cattr.register_unstructure_hook_factory(
    lambda cls: lenient_issubclass(cls, LLMConfig),
    lambda cls: make_dict_unstructure_fn(cls, bentoml_cattr, _cattrs_omit_if_default=False),
)


def structure_llm_config(data: dict[str, t.Any], cls: type[LLMConfig]) -> LLMConfig:
    """
    Structure a dictionary to a LLMConfig object.

    Essentially, if the given dictionary contains a 'generation_config' key, then we will
    use it for LLMConfig.generation_config

    Otherwise, we will filter out all keys are first in LLMConfig, parse it in, then
    parse the remaining keys into LLMConfig.generation_config
    """
    if not LazyType(DictStrAny).isinstance(data):
        raise RuntimeError(f"Expected a dictionary, but got {type(data)}")

    cls_attrs = {k: v for k, v in data.items() if k in cls.__openllm_attrs__}
    if "generation_config" in data:
        generation_config = data.pop("generation_config")
        if not LazyType(DictStrAny).isinstance(generation_config):
            raise RuntimeError(f"Expected a dictionary, but got {type(generation_config)}")
    else:
        generation_config = {k: v for k, v in data.items() if k in attr.fields_dict(cls.generation_class)}
    not_extras = list(cls_attrs) + list(generation_config)
    # The rest should be passed to extras
    data = {k: v for k, v in data.items() if k not in not_extras}

    return cls(generation_config=generation_config, __openllm_extras__=data, **cls_attrs)


bentoml_cattr.register_structure_hook_func(lambda cls: lenient_issubclass(cls, LLMConfig), structure_llm_config)
