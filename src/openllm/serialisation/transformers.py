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

import bentoml
from bentoml._internal.frameworks.transformers import make_default_signatures
from bentoml._internal.models.model import ModelOptions

from .constants import FRAMEWORK_TO_AUTOCLASS_MAPPING
from .constants import HUB_ATTRS
from ..exceptions import OpenLLMException
from ..utils import LazyLoader
from ..utils import LazyType
from ..utils import device_count
from ..utils import first_not_none
from ..utils import generate_context
from ..utils import generate_labels
from ..utils import is_autogptq_available
from ..utils import is_torch_available
from ..utils import normalize_attrs_to_model_tokenizer_pair


if t.TYPE_CHECKING:
    import auto_gptq as autogptq
    import torch
    import vllm

    import openllm
    import transformers as _transformers
    from transformers.models.auto.auto_factory import _BaseAutoModelClass

    from .._llm import M
    from .._llm import T
    from .._types import DictStrAny
else:
    vllm = LazyLoader("vllm", globals(), "vllm")
    autogptq = LazyLoader("autogptq", globals(), "auto_gptq")
    _transformers = LazyLoader("_transformers", globals(), "transformers")
    torch = LazyLoader("torch", globals(), "torch")

_object_setattr = object.__setattr__

def process_transformers_config(model_id: str, trust_remote_code: bool, **attrs: t.Any) -> tuple[_transformers.PretrainedConfig, dict[str, t.Any], dict[str, t.Any]]:
    """Process transformers config and return PretrainedConfig with hub_kwargs and the rest of kwargs."""
    config: _transformers.PretrainedConfig | None = attrs.pop("config", None)
    # this logic below is synonymous to handling `from_pretrained` attrs.
    hub_attrs = {k: attrs.pop(k) for k in HUB_ATTRS if k in attrs}
    if not isinstance(config, _transformers.PretrainedConfig):
        copied_attrs = copy.deepcopy(attrs)
        if copied_attrs.get("torch_dtype", None) == "auto": copied_attrs.pop("torch_dtype")
        config, attrs = _transformers.AutoConfig.from_pretrained(model_id, return_unused_kwargs=True, trust_remote_code=trust_remote_code, **hub_attrs, **copied_attrs)
    return t.cast("_transformers.PretrainedConfig", config), hub_attrs, t.cast("dict[str, t.Any]", attrs)

def infer_tokenizers_class_for_llm(__llm: openllm.LLM[t.Any, T]) -> T:
    tokenizer_class = __llm.config["tokenizer_class"]
    if tokenizer_class is None: tokenizer_class = "AutoTokenizer"
    __cls = getattr(_transformers, tokenizer_class)
    if __cls is None: raise ValueError(f"{tokenizer_class} is not a valid Tokenizer class from 'transformers.' Set '{__llm}.__config__[\"trust_remote_code\"] = True' and try again.")
    return __cls

def infer_autoclass_from_llm_config(llm: openllm.LLM[M, T], config: _transformers.PretrainedConfig) -> _BaseAutoModelClass:
    if llm.config["trust_remote_code"]:
        autoclass = "AutoModelForSeq2SeqLM" if llm.config["model_type"] == "seq2seq_lm" else "AutoModelForCausalLM"
        if not hasattr(config, "auto_map"): raise ValueError(f"Invalid configuraiton for {llm.model_id}. ``trust_remote_code=True`` requires `transformers.PretrainedConfig` to contain a `auto_map` mapping")
        # in case this model doesn't use the correct auto class for model type, for example like chatglm
        # where it uses AutoModel instead of AutoModelForCausalLM. Then we fallback to AutoModel
        if autoclass not in config.auto_map: autoclass = "AutoModel"
        return getattr(_transformers, autoclass)
    else:
        if type(config) in _transformers.MODEL_FOR_CAUSAL_LM_MAPPING: idx = 0
        elif type(config) in _transformers.MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING: idx = 1
        else: raise OpenLLMException(f"Model type {type(config)} is not supported yet.")
        return getattr(_transformers, FRAMEWORK_TO_AUTOCLASS_MAPPING[llm.__llm_implementation__][idx])

def import_model(llm: openllm.LLM[M, T], *decls: t.Any, trust_remote_code: bool, **attrs: t.Any) -> bentoml.Model:
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
    _, tokenizer_attrs = llm.llm_parameters
    quantize_method = llm._quantize_method
    safe_serialisation = first_not_none(attrs.get("safe_serialization"), default=llm._serialisation_format == "safetensors")
    # Disable safe serialization with vLLM
    if llm.__llm_implementation__ == "vllm": safe_serialisation = False
    metadata: DictStrAny = {"safe_serialisation": safe_serialisation, "_quantize": quantize_method if quantize_method is not None else False}
    signatures: DictStrAny = {}
    if quantize_method == "gptq":
        if not is_autogptq_available(): raise OpenLLMException("GPTQ quantisation requires 'auto-gptq' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\"'")
        if llm.config["model_type"] != "causal_lm": raise OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
        model = autogptq.AutoGPTQForCausalLM.from_quantized(
            llm.model_id,
            *decls,
            quantize_config=t.cast("autogptq.BaseQuantizeConfig", llm.quantization_config),
            trust_remote_code=trust_remote_code,
            use_safetensors=safe_serialisation,
            **hub_attrs,
            **attrs,
        )
        metadata.update({"_pretrained_class": model.__class__.__name__, "_framework": model.model.framework})
        signatures["generate"] = {"batchable": False}
    else:
        # this model might be called with --quantize int4, therefore we need to pop this out
        # since saving int4 is not yet supported
        if "quantization_config" in attrs and getattr(attrs["quantization_config"], "load_in_4bit", False): attrs.pop("quantization_config")
        model = infer_autoclass_from_llm_config(llm, config).from_pretrained(
            llm.model_id,
            *decls,
            config=config,
            trust_remote_code=trust_remote_code,
            **hub_attrs,
            **attrs,
        )
        metadata.update({"_pretrained_class": model.__class__.__name__, "_framework": model.framework})

    _tokenizer = infer_tokenizers_class_for_llm(llm).from_pretrained(
        llm.model_id,
        trust_remote_code=trust_remote_code,
        **hub_attrs,
        **tokenizer_attrs,
    )
    if _tokenizer.pad_token is None: _tokenizer.pad_token = _tokenizer.eos_token

    # NOTE: quick hack to set the loaded into llm object to use with save_pretrained
    # to avoid recursive call when the model is not yet available in local store
    _object_setattr(llm, "__llm_model__", model)
    _object_setattr(llm, "__llm_tokenizer__", _tokenizer)

    try:
        with bentoml.models.create(
            llm.tag,
            module="openllm.serialisation.transformers",
            api_version="v1",
            context=generate_context(framework_name="openllm"),
            labels=generate_labels(llm),
            signatures=signatures if signatures else make_default_signatures(model),
            options=ModelOptions(),
            external_modules=[importlib.import_module(model.__module__), importlib.import_module(_tokenizer.__module__)] if trust_remote_code else None,
            metadata=metadata,
        ) as bentomodel:
            save_pretrained(llm, bentomodel.path, safe_serialization=safe_serialisation)
            return bentomodel
    finally:
        # NOTE: We need to free up the cache after importing the model
        # in the case where users first run openllm start without the model
        # available locally.
        if is_torch_available() and torch.cuda.is_available(): torch.cuda.empty_cache()

def get(llm: openllm.LLM[M, T], auto_import: bool = False) -> bentoml.Model:
    """Return an instance of ``bentoml.Model`` from given LLM instance.

    By default, it will try to check the model in the local store.
    If model is not found, and ``auto_import`` is set to True, it will try to import the model from HuggingFace Hub.

    Otherwise, it will raises a ``bentoml.exceptions.NotFound``.
    """
    try:
        model = bentoml.models.get(llm.tag)
        # compat with bentoml.transformers.get
        if model.info.module not in ("openllm.serialisation.transformers", "bentoml.transformers", "bentoml._internal.frameworks.transformers", __name__):
            raise bentoml.exceptions.NotFound(f"Model {model.tag} was saved with module {model.info.module}, not loading with 'openllm.serialisation.transformers'.")
        if "runtime" in model.info.labels and model.info.labels["runtime"] != llm.runtime:
            raise OpenLLMException(f"Model {model.tag} was saved with runtime {model.info.labels['runtime']}, not loading with {llm.runtime}.")
        return model
    except bentoml.exceptions.NotFound:
        if auto_import: return import_model(llm, trust_remote_code=llm.__llm_trust_remote_code__)
        raise

def load_model(llm: openllm.LLM[M, T], *decls: t.Any, **attrs: t.Any) -> M:
    """Load the model from BentoML store.

    By default, it will try to find check the model in the local store.
    If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
    """
    config, hub_attrs, attrs = process_transformers_config(llm.model_id, llm.__llm_trust_remote_code__, **attrs)
    safe_serialization = first_not_none(t.cast(t.Optional[bool], llm._bentomodel.info.metadata.get("safe_serialisation", None)), attrs.pop("safe_serialization", None), default=llm._serialisation_format == "safetensors")
    if "_quantize" in llm._bentomodel.info.metadata and llm._bentomodel.info.metadata["_quantize"] == "gptq":
        if not is_autogptq_available(): raise OpenLLMException("GPTQ quantisation requires 'auto-gptq' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\"'")
        if llm.config["model_type"] != "causal_lm": raise OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
        return autogptq.AutoGPTQForCausalLM.from_quantized(
            llm._bentomodel.path,
            *decls,
            quantize_config=t.cast("autogptq.BaseQuantizeConfig", llm.quantization_config),
            trust_remote_code=llm.__llm_trust_remote_code__,
            use_safetensors=safe_serialization,
            **hub_attrs,
            **attrs,
        )

    model = infer_autoclass_from_llm_config(llm, config).from_pretrained(
        llm._bentomodel.path,
        *decls,
        config=config,
        trust_remote_code=llm.__llm_trust_remote_code__,
        **hub_attrs,
        **attrs,
    )
    # NOTE: we only cast and load the model if it is not already quantized and setup correctly
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_quantized", False)
    if torch.cuda.is_available() and device_count() == 1 and not loaded_in_kbit:
        try: model = model.to("cuda")
        except torch.cuda.OutOfMemoryError as err: raise RuntimeError(f"Failed to convert {llm.config['model_name']} with model_id '{llm.model_id}' to CUDA.\nNote: You can try out '--quantize int8 | int4' for dynamic quantization.") from err
    if llm.bettertransformer and llm.__llm_implementation__ == "pt" and not isinstance(model, _transformers.Pipeline):
        # BetterTransformer is currently only supported on PyTorch.
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
    return t.cast("M", model)

def save_pretrained(
    llm: openllm.LLM[M, T],
    save_directory: str,
    is_main_process: bool = True,
    state_dict: DictStrAny | None = None,
    save_function: t.Callable[..., None] | None = None,
    push_to_hub: bool = False,
    max_shard_size: int | str = "2GB",
    safe_serialization: bool = False,
    variant: str | None = None,
    **attrs: t.Any,
) -> None:
    """Light wrapper around ``transformers.PreTrainedTokenizer.save_pretrained`` and ``transformers.PreTrainedModel.save_pretrained``."""
    save_function = first_not_none(save_function, default=torch.save)
    model_save_attrs, tokenizer_save_attrs = normalize_attrs_to_model_tokenizer_pair(**attrs)
    safe_serialization = safe_serialization or llm._serialisation_format == "safetensors"
    # NOTE: disable safetensors for vllm
    if llm.__llm_implementation__ == "vllm": safe_serialization = False
    if llm._quantize_method == "gptq":
        if not is_autogptq_available(): raise OpenLLMException("GPTQ quantisation requires 'auto-gptq' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\"'")
        if llm.config["model_type"] != "causal_lm": raise OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
        llm.model.save_quantized(save_directory, use_safetensors=safe_serialization)
    elif LazyType["vllm.LLMEngine"]("vllm.LLMEngine").isinstance(llm.model): raise RuntimeError("vllm.LLMEngine cannot be serialisation directly. This happens when 'save_pretrained' is called directly after `openllm.AutoVLLM` is initialized.")
    elif isinstance(llm.model, _transformers.Pipeline): llm.model.save_pretrained(save_directory, safe_serialization=safe_serialization)
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
            **model_save_attrs
        )
    llm.tokenizer.save_pretrained(save_directory, push_to_hub=push_to_hub, **tokenizer_save_attrs)
