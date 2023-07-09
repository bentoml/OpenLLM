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
"""Serialisation related implementation for Transformers-based implementation."""

from __future__ import annotations
import copy
import importlib
import typing as t

import cloudpickle

import bentoml
from bentoml._internal.frameworks.transformers import make_default_signatures
from bentoml._internal.models.model import CUSTOM_OBJECTS_FILENAME
from bentoml._internal.models.model import ModelOptions

from .constants import FRAMEWORK_TO_AUTOCLASS_MAPPING
from .constants import MODEL_TO_AUTOCLASS_MAPPING
from ..exceptions import OpenLLMException
from ..utils import LazyLoader
from ..utils import generate_context
from ..utils import generate_labels
from ..utils import is_torch_available
from ..utils import normalize_attrs_to_model_tokenizer_pair


if t.TYPE_CHECKING:
    import torch

    import openllm
    import transformers
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    from .._llm import _M
    from .._llm import _T
    from .._types import DictStrAny
    from .._types import ModelProtocol
    from .._types import P
    from .._types import TokenizerProtocol
else:
    transformers = LazyLoader("transformers", globals(), "transformers")
    torch = LazyLoader("torch", globals(), "torch")


def process_transformers_config(
    model_id: str, trust_remote_code: bool, **attrs: t.Any
) -> tuple[transformers.PretrainedConfig, dict[str, t.Any], dict[str, t.Any]]:
    """Process transformers config and return PretrainedConfig with hub_kwargs and the rest of kwargs."""
    config: transformers.PretrainedConfig = attrs.pop("config", None)

    # this logic below is synonymous to handling `from_pretrained` attrs.
    hub_kwds_names = [
        "cache_dir",
        "code_revision",
        "force_download",
        "local_files_only",
        "proxies",
        "resume_download",
        "revision",
        "subfolder",
        "use_auth_token",
    ]
    hub_attrs = {k: attrs.pop(k) for k in hub_kwds_names if k in attrs}
    if not isinstance(config, transformers.PretrainedConfig):
        copied_attrs = copy.deepcopy(attrs)
        if copied_attrs.get("torch_dtype", None) == "auto":
            copied_attrs.pop("torch_dtype")
        config, attrs = t.cast(
            "tuple[transformers.PretrainedConfig, dict[str, t.Any]]",
            transformers.AutoConfig.from_pretrained(
                model_id, return_unused_kwargs=True, trust_remote_code=trust_remote_code, **hub_attrs, **copied_attrs
            ),
        )
    return config, hub_attrs, attrs


def infer_autoclass_from_llm_config(
    llm: openllm.LLM[t.Any, t.Any], config: transformers.PretrainedConfig
) -> _BaseAutoModelClass:
    if llm.config["model_name"] in MODEL_TO_AUTOCLASS_MAPPING:
        return getattr(transformers, MODEL_TO_AUTOCLASS_MAPPING[llm.config["model_name"]][llm.__llm_implementation__])
    else:
        if type(config) in transformers.MODEL_FOR_CAUSAL_LM_MAPPING:
            idx = 0
        elif type(config) in transformers.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING:
            idx = 1
        else:
            raise OpenLLMException(f"Model type {type(config)} is not supported yet.")

        return getattr(transformers, FRAMEWORK_TO_AUTOCLASS_MAPPING[llm.__llm_implementation__][idx])


def import_model(
    llm: openllm.LLM[t.Any, t.Any],
    *decls: t.Any,
    trust_remote_code: bool,
    **attrs: t.Any,
) -> bentoml.Model:
    """Auto detect model type from given model_id and import it to bentoml's model store.

    For all kwargs, it will be parsed into `transformers.AutoConfig.from_pretrained` first,
    returning all of the unused kwargs.
    The unused kwargs then parsed directly into AutoModelForSeq2SeqLM or AutoModelForCausalLM (+ TF, Flax variants).
    For all tokenizer kwargs, make sure to prefix it with `_tokenizer_` to avoid confusion.

    Note: Currently, there are only two tasks supported: `text-generation` and `text2text-generation`.

    Refer to Transformers documentation for more information about kwargs.

    Args:
        llm: The LLM instance for this given model.
        trust_remote_code: Whether to trust the remote code when loading the model.
        *decls: Args to be passed into AutoModelForSeq2SeqLM or AutoModelForCausalLM (+ TF, Flax variants).
        **attrs: Kwargs to be passed into AutoModelForSeq2SeqLM or AutoModelForCausalLM (+ TF, Flax variants).
    """
    config, hub_attrs, attrs = process_transformers_config(llm.model_id, trust_remote_code, **attrs)

    # NOTE: get the base args and attrs, then
    # allow override via import_model
    (model_decls, model_attrs), tokenizer_attrs = llm.llm_parameters
    decls = (*model_decls, *decls)
    attrs = {**model_attrs, **attrs}

    tokenizer = t.cast(
        "transformers.PreTrainedTokenizer",
        transformers.AutoTokenizer.from_pretrained(
            llm.model_id,
            config=config,
            trust_remote_code=trust_remote_code,
            **hub_attrs,
            **tokenizer_attrs,
        ),
    )

    model = t.cast(
        "transformers.PreTrainedModel",
        infer_autoclass_from_llm_config(llm, config).from_pretrained(
            llm.model_id,
            *decls,
            config=config,
            trust_remote_code=trust_remote_code,
            **hub_attrs,
            **attrs,
        ),
    )

    try:
        with bentoml.models.create(
            llm.tag,
            module="openllm.serialisation.transformers",
            api_version="v1",
            context=generate_context(framework_name="openllm"),
            labels=generate_labels(llm),
            options=ModelOptions(),
            signatures=make_default_signatures(model),
            external_modules=[
                importlib.import_module(model.__module__),
                importlib.import_module(tokenizer.__module__),
            ],
            metadata={
                "_pretrained_class": model.__class__.__name__,
                "_framework": model.framework,
            },
        ) as bentomodel:
            model.save_pretrained(bentomodel.path)
            tokenizer.save_pretrained(bentomodel.path)
            return bentomodel
    finally:
        # NOTE: We need to free up the cache after importing the model
        # in the case where users first run openllm start without the model
        # available locally.
        if is_torch_available() and torch.cuda.is_available():
            torch.cuda.empty_cache()


def get(llm: openllm.LLM[t.Any, t.Any], auto_import: bool = False) -> bentoml.Model:
    """Return an instance of ``bentoml.Model`` from given LLM instance.

    By default, it will try to check the model in the local store.
    If model is not found, and ``auto_import`` is set to True, it will try to import the model from HuggingFace Hub.

    Otherwise, it will raises a ``bentoml.exceptions.NotFound``.
    """
    try:
        model = bentoml.models.get(llm.tag)
        if model.info.module not in (
            "openllm.serialisation.transformers",
            # compat with bentoml.transformers.get
            "bentoml.transformers",
            "bentoml._internal.frameworks.transformers",
            __name__,
        ):
            raise bentoml.exceptions.NotFound(
                f"Model {model.tag} was saved with module {model.info.module}, not loading with 'openllm.serialisation.transformers'."
            )
        if "runtime" in model.info.labels and model.info.labels["runtime"] != llm.runtime:
            raise OpenLLMException(
                f"Model {model.tag} was saved with runtime {model.info.labels['runtime']}, not loading with {llm.runtime}."
            )
        return model
    except bentoml.exceptions.NotFound:
        if auto_import:
            return import_model(llm, trust_remote_code=llm.__llm_trust_remote_code__)
        raise


def load_model(llm: openllm.LLM[_M, t.Any], *decls: t.Any, **attrs: t.Any) -> ModelProtocol[_M]:
    """Load the model from BentoML store.

    By default, it will try to find check the model in the local store.
    If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
    """
    config, hub_attrs, attrs = process_transformers_config(llm.model_id, llm.__llm_trust_remote_code__, **attrs)

    # NOTE: get the base args and attrs, then
    # allow override via import_model
    (model_decls, model_attrs), _ = llm.llm_parameters
    decls = (*model_decls, *decls)
    attrs = {**model_attrs, **attrs}

    if llm.__llm_custom_load__:
        model = llm.load_model(llm.tag, *decls, **hub_attrs, **attrs)
    else:
        model = infer_autoclass_from_llm_config(llm, config).from_pretrained(
            llm._bentomodel.path, *decls, config=config, **hub_attrs, **attrs
        )
    if llm.bettertransformer and llm.__llm_implementation__ == "pt" and not isinstance(model, transformers.Pipeline):
        # BetterTransformer is currently only supported on PyTorch.
        from optimum.bettertransformer import BetterTransformer

        model = BetterTransformer.transform(model)
    return t.cast("ModelProtocol[_M]", model)


def load_tokenizer(llm: openllm.LLM[t.Any, _T]) -> TokenizerProtocol[_T]:
    """Load the tokenizer from BentoML store.

    By default, it will try to find the bentomodel whether it is in store..
    If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
    """
    (_, _), tokenizer_attrs = llm.llm_parameters
    if llm.__llm_custom_tokenizer__:
        tokenizer = llm.load_tokenizer(llm.tag, **tokenizer_attrs)
    else:
        bentomodel_fs = llm._bentomodel._fs
        if bentomodel_fs.isfile(CUSTOM_OBJECTS_FILENAME):
            with bentomodel_fs.open(CUSTOM_OBJECTS_FILENAME, "rb") as cofile:
                try:
                    tokenizer = cloudpickle.load(t.cast("t.IO[bytes]", cofile))["tokenizer"]
                except KeyError:
                    # This could happen if users implement their own import_model
                    raise OpenLLMException(
                        "Model does not have tokenizer. Make sure to save \
                        the tokenizer within the model via 'custom_objects'.\
                        For example: bentoml.transformers.save_model(..., custom_objects={'tokenizer': tokenizer}))"
                    ) from None
        else:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                bentomodel_fs.getsyspath("/"),
                trust_remote_code=llm.__llm_trust_remote_code__,
                **tokenizer_attrs,
            )
    return t.cast("TokenizerProtocol[_T]", tokenizer)


def save_pretrained(
    llm: openllm.LLM[t.Any, t.Any],
    save_directory: str,
    is_main_process: bool = True,
    state_dict: DictStrAny | None = None,
    save_function: t.Callable[P, None] | None = None,
    push_to_hub: bool = False,
    max_shard_size: int | str = "10GB",
    safe_serialization: bool = False,
    variant: str | None = None,
    **attrs: t.Any,
):
    """Light wrapper around ``transformers.PreTrainedTokenizer.save_pretrained`` and ``transformers.PreTrainedModel.save_pretrained``."""
    if save_function is None:
        save_function = torch.save

    model_save_attrs, tokenizer_save_attrs = normalize_attrs_to_model_tokenizer_pair(**attrs)

    if isinstance(llm.model, transformers.Pipeline):
        llm.model.save_pretrained(save_directory, safe_serialization=safe_serialization)
    else:
        llm.model.save_pretrained(
            save_directory,
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=safe_serialization,
            variant=variant,
            **model_save_attrs,
        )
    llm.tokenizer.save_pretrained(save_directory, push_to_hub=push_to_hub, **tokenizer_save_attrs)
