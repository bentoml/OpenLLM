# mypy: disable-error-code="name-defined,misc"
"""Serialisation related implementation for Transformers-based implementation."""
from __future__ import annotations
import importlib
import logging
import typing as t

from huggingface_hub import snapshot_download
from simple_di import Provide, inject

import bentoml
import openllm
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.model import ModelOptions
from openllm.serialisation.transformers.weights import HfIgnore

from ._helpers import (
  check_unintialised_params,
  infer_autoclass_from_llm,
  infer_tokenizers_from_llm,
  make_model_signatures,
  process_config,
  update_model,
)

if t.TYPE_CHECKING:
  import types

  import torch.nn

  from bentoml._internal.models import ModelStore
  from openllm._llm import M, T
  from openllm._types import DictStrAny

vllm = openllm.utils.LazyLoader("vllm", globals(), "vllm")
autogptq = openllm.utils.LazyLoader("autogptq", globals(), "auto_gptq")
transformers = openllm.utils.LazyLoader("transformers", globals(), "transformers")
torch = openllm.utils.LazyLoader("torch", globals(), "torch")

logger = logging.getLogger(__name__)

__all__ = ["import_model", "get", "load_model", "save_pretrained"]

@inject
def import_model(llm: openllm.LLM[M, T], *decls: t.Any, trust_remote_code: bool, _model_store: ModelStore = Provide[BentoMLContainer.model_store], **attrs: t.Any) -> bentoml.Model:
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
  config, hub_attrs, attrs = process_config(llm.model_id, trust_remote_code, **attrs)
  _, tokenizer_attrs = llm.llm_parameters
  quantize_method = llm._quantize_method
  safe_serialisation = openllm.utils.first_not_none(attrs.get("safe_serialization"), default=llm._serialisation_format == "safetensors")
  # Disable safe serialization with vLLM
  if llm.__llm_implementation__ == "vllm": safe_serialisation = False
  metadata: DictStrAny = {"safe_serialisation": safe_serialisation, "_quantize": quantize_method is not None and quantize_method}
  signatures: DictStrAny = {}

  if quantize_method == "gptq":
    if not openllm.utils.is_autogptq_available(): raise openllm.exceptions.OpenLLMException("GPTQ quantisation requires 'auto-gptq' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\"'")
    if llm.config["model_type"] != "causal_lm": raise openllm.exceptions.OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
    signatures["generate"] = {"batchable": False}
  else:
    # this model might be called with --quantize int4, therefore we need to pop this out
    # since saving int4 is not yet supported
    if "quantization_config" in attrs and getattr(attrs["quantization_config"], "load_in_4bit", False): attrs.pop("quantization_config")
    if llm.__llm_implementation__ != "flax": attrs["use_safetensors"] = safe_serialisation
    metadata["_framework"] = "pt" if llm.__llm_implementation__ == "vllm" else llm.__llm_implementation__

  tokenizer = infer_tokenizers_from_llm(llm).from_pretrained(llm.model_id, trust_remote_code=trust_remote_code, **hub_attrs, **tokenizer_attrs)
  if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

  external_modules: list[types.ModuleType] = [importlib.import_module(tokenizer.__module__)]
  imported_modules: list[types.ModuleType] = []
  bentomodel = bentoml.Model.create(llm.tag, module="openllm.serialisation.transformers", api_version="v1", options=ModelOptions(), context=openllm.utils.generate_context(framework_name="openllm"), labels=openllm.utils.generate_labels(llm), signatures=signatures if signatures else make_model_signatures(llm))
  with openllm.utils.analytics.set_bentoml_tracking():
    try:
      bentomodel.enter_cloudpickle_context(external_modules, imported_modules)
      tokenizer.save_pretrained(bentomodel.path)
      if quantize_method == "gptq":
        if not openllm.utils.is_autogptq_available(): raise openllm.exceptions.OpenLLMException("GPTQ quantisation requires 'auto-gptq' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\"'")
        if llm.config["model_type"] != "causal_lm": raise openllm.exceptions.OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
        logger.debug("Saving model with GPTQ quantisation will require loading model into memory.")
        model = autogptq.AutoGPTQForCausalLM.from_quantized(llm.model_id, *decls, quantize_config=t.cast("autogptq.BaseQuantizeConfig", llm.quantization_config), trust_remote_code=trust_remote_code, use_safetensors=safe_serialisation, **hub_attrs, **attrs,)
        update_model(bentomodel, metadata={"_pretrained_class": model.__class__.__name__, "_framework": model.model.framework})
        model.save_quantized(bentomodel.path, use_safetensors=safe_serialisation)
      else:
        architectures = getattr(config, "architectures", [])
        if not architectures:
          raise RuntimeError("Failed to determine the architecture for this model. Make sure the `config.json` is valid and can be loaded with `transformers.AutoConfig`")
        architecture = architectures[0]
        update_model(bentomodel, metadata={"_pretrained_class": architecture})
        if llm._local:
          # possible local path
          logger.debug("Model will be loaded into memory to save to target store as it is from local path.")
          model = infer_autoclass_from_llm(llm, config).from_pretrained(llm.model_id, *decls, config=config, trust_remote_code=trust_remote_code, **hub_attrs, **attrs)
          # for trust_remote_code to work
          bentomodel.enter_cloudpickle_context([importlib.import_module(model.__module__)], imported_modules)
          model.save_pretrained(bentomodel.path, max_shard_size="5GB", safe_serialization=safe_serialisation)
        else:
          # we will clone the all tings into the bentomodel path without loading model into memory
          snapshot_download(llm.model_id, local_dir=bentomodel.path, local_dir_use_symlinks=False, ignore_patterns=HfIgnore.ignore_patterns(llm))
    except Exception:
      raise
    else:
      bentomodel.flush()  # type: ignore[no-untyped-call]
      bentomodel.save(_model_store)
      openllm.utils.analytics.track(openllm.utils.analytics.ModelSaveEvent(module=bentomodel.info.module, model_size_in_kb=openllm.utils.calc_dir_size(bentomodel.path) / 1024))
    finally:
      bentomodel.exit_cloudpickle_context(imported_modules)
      # NOTE: We need to free up the cache after importing the model
      # in the case where users first run openllm start without the model available locally.
      if openllm.utils.is_torch_available() and torch.cuda.is_available(): torch.cuda.empty_cache()

    return bentomodel

def get(llm: openllm.LLM[M, T], auto_import: bool = False) -> bentoml.Model:
  """Return an instance of ``bentoml.Model`` from given LLM instance.

  By default, it will try to check the model in the local store.
  If model is not found, and ``auto_import`` is set to True, it will try to import the model from HuggingFace Hub.

  Otherwise, it will raises a ``bentoml.exceptions.NotFound``.
  """
  try:
    model = bentoml.models.get(llm.tag)
    if model.info.module not in ("openllm.serialisation.transformers", __name__):
      raise bentoml.exceptions.NotFound(f"Model {model.tag} was saved with module {model.info.module}, not loading with 'openllm.serialisation.transformers'.")
    if "runtime" in model.info.labels and model.info.labels["runtime"] != llm.runtime:
      raise openllm.exceptions.OpenLLMException(f"Model {model.tag} was saved with runtime {model.info.labels['runtime']}, not loading with {llm.runtime}.")
    return model
  except bentoml.exceptions.NotFound as err:
    if auto_import: return import_model(llm, trust_remote_code=llm.__llm_trust_remote_code__)
    raise err from None

def load_model(llm: openllm.LLM[M, T], *decls: t.Any, **attrs: t.Any) -> M:
  """Load the model from BentoML store.

  By default, it will try to find check the model in the local store.
  If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
  """
  config, hub_attrs, attrs = process_config(llm.model_id, llm.__llm_trust_remote_code__, **attrs)
  safe_serialization = openllm.utils.first_not_none(t.cast(t.Optional[bool], llm._bentomodel.info.metadata.get("safe_serialisation", None)), attrs.pop("safe_serialization", None), default=llm._serialisation_format == "safetensors")
  if "_quantize" in llm._bentomodel.info.metadata and llm._bentomodel.info.metadata["_quantize"] == "gptq":
    if not openllm.utils.is_autogptq_available(): raise openllm.exceptions.OpenLLMException("GPTQ quantisation requires 'auto-gptq' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\"'")
    if llm.config["model_type"] != "causal_lm": raise openllm.exceptions.OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
    return autogptq.AutoGPTQForCausalLM.from_quantized(llm._bentomodel.path, *decls, quantize_config=t.cast("autogptq.BaseQuantizeConfig", llm.quantization_config), trust_remote_code=llm.__llm_trust_remote_code__, use_safetensors=safe_serialization, **hub_attrs, **attrs)

  device_map = attrs.pop("device_map", "auto" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None)
  model = infer_autoclass_from_llm(llm, config).from_pretrained(llm._bentomodel.path, *decls, config=config, trust_remote_code=llm.__llm_trust_remote_code__, device_map=device_map, **hub_attrs, **attrs).eval()
  # BetterTransformer is currently only supported on PyTorch.
  if llm.bettertransformer and isinstance(model, transformers.PreTrainedModel): model = model.to_bettertransformer()
  if llm.__llm_implementation__ in {"pt", "vllm"}: check_unintialised_params(model)
  return t.cast("M", model)

def save_pretrained(llm: openllm.LLM[M, T], save_directory: str, is_main_process: bool = True, state_dict: DictStrAny | None = None, save_function: t.Any | None = None, push_to_hub: bool = False, max_shard_size: int | str = "10GB", safe_serialization: bool = False, variant: str | None = None, **attrs: t.Any) -> None:
  save_function = t.cast(t.Callable[..., None], openllm.utils.first_not_none(save_function, default=torch.save))
  model_save_attrs, tokenizer_save_attrs = openllm.utils.normalize_attrs_to_model_tokenizer_pair(**attrs)
  safe_serialization = safe_serialization or llm._serialisation_format == "safetensors"
  # NOTE: disable safetensors for vllm
  if llm.__llm_implementation__ == "vllm": safe_serialization = False
  if llm._quantize_method == "gptq":
    if not openllm.utils.is_autogptq_available(): raise openllm.exceptions.OpenLLMException("GPTQ quantisation requires 'auto-gptq' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\"'")
    if llm.config["model_type"] != "causal_lm": raise openllm.exceptions.OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
    if not openllm.utils.lenient_issubclass(llm.model, autogptq.modeling.BaseGPTQForCausalLM): raise ValueError(f"Model is not a BaseGPTQForCausalLM (type: {type(llm.model)})")
    t.cast("autogptq.modeling.BaseGPTQForCausalLM", llm.model).save_quantized(save_directory, use_safetensors=safe_serialization)
  elif openllm.utils.LazyType["vllm.LLMEngine"]("vllm.LLMEngine").isinstance(llm.model):
    raise RuntimeError("vllm.LLMEngine cannot be serialisation directly. This happens when 'save_pretrained' is called directly after `openllm.AutoVLLM` is initialized.")
  elif isinstance(llm.model, transformers.Pipeline):
    llm.model.save_pretrained(save_directory, safe_serialization=safe_serialization)
  else:
    # We can safely cast here since it will be the PreTrainedModel protocol.
    t.cast("transformers.PreTrainedModel", llm.model).save_pretrained(save_directory, is_main_process=is_main_process, state_dict=state_dict, save_function=save_function, push_to_hub=push_to_hub, max_shard_size=max_shard_size, safe_serialization=safe_serialization, variant=variant, **model_save_attrs)
  llm.tokenizer.save_pretrained(save_directory, push_to_hub=push_to_hub, **tokenizer_save_attrs)
