'''Serialisation related implementation for Transformers-based implementation.'''
from __future__ import annotations
import importlib
import logging
import typing as t

from huggingface_hub import snapshot_download
from simple_di import Provide
from simple_di import inject

import bentoml
import openllm

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.model import ModelOptions

from ._helpers import check_unintialised_params
from ._helpers import infer_autoclass_from_llm
from ._helpers import infer_tokenizers_from_llm
from ._helpers import make_model_signatures
from ._helpers import process_config
from ._helpers import update_model
from .weights import HfIgnore

if t.TYPE_CHECKING:
  import types

  import auto_gptq as autogptq
  import torch
  import torch.nn

  from bentoml._internal.models import ModelStore
  from openllm_core._typing_compat import DictStrAny
  from openllm_core._typing_compat import M
  from openllm_core._typing_compat import T
else:
  autogptq = openllm.utils.LazyLoader('autogptq', globals(), 'auto_gptq')
  torch = openllm.utils.LazyLoader('torch', globals(), 'torch')

logger = logging.getLogger(__name__)

__all__ = ['import_model', 'get', 'load_model']

@inject
def import_model(llm: openllm.LLM[M, T],
                 *decls: t.Any,
                 trust_remote_code: bool,
                 _model_store: ModelStore = Provide[BentoMLContainer.model_store],
                 **attrs: t.Any) -> bentoml.Model:
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
  safe_serialisation = openllm.utils.first_not_none(attrs.get('safe_serialization'),
                                                    default=llm._serialisation_format == 'safetensors')
  # Disable safe serialization with vLLM
  if llm.__llm_backend__ == 'vllm': safe_serialisation = False
  metadata: DictStrAny = {
      'safe_serialisation': safe_serialisation,
      '_quantize': quantize_method is not None and quantize_method
  }
  signatures: DictStrAny = {}

  if quantize_method == 'gptq':
    if not openllm.utils.is_autogptq_available():
      raise openllm.exceptions.OpenLLMException(
          "GPTQ quantisation requires 'auto-gptq' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\"'"
      )
    if llm.config['model_type'] != 'causal_lm':
      raise openllm.exceptions.OpenLLMException(
          f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
    signatures['generate'] = {'batchable': False}
  else:
    # this model might be called with --quantize int4, therefore we need to pop this out
    # since saving int4 is not yet supported
    if 'quantization_config' in attrs and getattr(attrs['quantization_config'], 'load_in_4bit', False):
      attrs.pop('quantization_config')
    if llm.__llm_backend__ != 'flax': attrs['use_safetensors'] = safe_serialisation
    metadata['_framework'] = 'pt' if llm.__llm_backend__ == 'vllm' else llm.__llm_backend__

  tokenizer = infer_tokenizers_from_llm(llm).from_pretrained(llm.model_id,
                                                             trust_remote_code=trust_remote_code,
                                                             **hub_attrs,
                                                             **tokenizer_attrs)
  if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

  external_modules: list[types.ModuleType] = [importlib.import_module(tokenizer.__module__)]
  imported_modules: list[types.ModuleType] = []
  bentomodel = bentoml.Model.create(llm.tag,
                                    module='openllm.serialisation.transformers',
                                    api_version='v1',
                                    options=ModelOptions(),
                                    context=openllm.utils.generate_context(framework_name='openllm'),
                                    labels=openllm.utils.generate_labels(llm),
                                    signatures=signatures if signatures else make_model_signatures(llm))
  with openllm.utils.analytics.set_bentoml_tracking():
    try:
      bentomodel.enter_cloudpickle_context(external_modules, imported_modules)
      tokenizer.save_pretrained(bentomodel.path)
      if quantize_method == 'gptq':
        if not openllm.utils.is_autogptq_available():
          raise openllm.exceptions.OpenLLMException(
              "GPTQ quantisation requires 'auto-gptq' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\"'"
          )
        if llm.config['model_type'] != 'causal_lm':
          raise openllm.exceptions.OpenLLMException(
              f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
        logger.debug('Saving model with GPTQ quantisation will require loading model into memory.')
        model = autogptq.AutoGPTQForCausalLM.from_quantized(llm.model_id,
                                                            *decls,
                                                            quantize_config=t.cast('autogptq.BaseQuantizeConfig',
                                                                                   llm.quantization_config),
                                                            trust_remote_code=trust_remote_code,
                                                            use_safetensors=safe_serialisation,
                                                            **hub_attrs,
                                                            **attrs)
        update_model(bentomodel,
                     metadata={
                         '_pretrained_class': model.__class__.__name__,
                         '_framework': model.model.framework
                     })
        model.save_quantized(bentomodel.path, use_safetensors=safe_serialisation)
      else:
        architectures = getattr(config, 'architectures', [])
        if not architectures:
          raise RuntimeError(
              'Failed to determine the architecture for this model. Make sure the `config.json` is valid and can be loaded with `transformers.AutoConfig`'
          )
        architecture = architectures[0]
        update_model(bentomodel, metadata={'_pretrained_class': architecture})
        if llm._local:
          # possible local path
          logger.debug('Model will be loaded into memory to save to target store as it is from local path.')
          model = infer_autoclass_from_llm(llm, config).from_pretrained(llm.model_id,
                                                                        *decls,
                                                                        config=config,
                                                                        trust_remote_code=trust_remote_code,
                                                                        **hub_attrs,
                                                                        **attrs)
          # for trust_remote_code to work
          bentomodel.enter_cloudpickle_context([importlib.import_module(model.__module__)], imported_modules)
          model.save_pretrained(bentomodel.path, max_shard_size='5GB', safe_serialization=safe_serialisation)
        else:
          # we will clone the all tings into the bentomodel path without loading model into memory
          snapshot_download(llm.model_id,
                            local_dir=bentomodel.path,
                            local_dir_use_symlinks=False,
                            ignore_patterns=HfIgnore.ignore_patterns(llm))
    except Exception:
      raise
    else:
      bentomodel.flush()  # type: ignore[no-untyped-call]
      bentomodel.save(_model_store)
      openllm.utils.analytics.track(
          openllm.utils.analytics.ModelSaveEvent(module=bentomodel.info.module,
                                                 model_size_in_kb=openllm.utils.calc_dir_size(bentomodel.path) / 1024))
    finally:
      bentomodel.exit_cloudpickle_context(imported_modules)
      # NOTE: We need to free up the cache after importing the model
      # in the case where users first run openllm start without the model available locally.
      if openllm.utils.is_torch_available() and torch.cuda.is_available(): torch.cuda.empty_cache()
    return bentomodel

def get(llm: openllm.LLM[M, T], auto_import: bool = False) -> bentoml.Model:
  '''Return an instance of ``bentoml.Model`` from given LLM instance.

  By default, it will try to check the model in the local store.
  If model is not found, and ``auto_import`` is set to True, it will try to import the model from HuggingFace Hub.

  Otherwise, it will raises a ``bentoml.exceptions.NotFound``.
  '''
  try:
    model = bentoml.models.get(llm.tag)
    if model.info.module not in ('openllm.serialisation.transformers'
                                 'bentoml.transformers', 'bentoml._internal.frameworks.transformers',
                                 __name__):  # NOTE: backward compatible with previous version of OpenLLM.
      raise bentoml.exceptions.NotFound(
          f"Model {model.tag} was saved with module {model.info.module}, not loading with 'openllm.serialisation.transformers'."
      )
    if 'backend' in model.info.labels and model.info.labels['backend'] != llm.__llm_backend__:
      raise openllm.exceptions.OpenLLMException(
          f"Model {model.tag} was saved with backend {model.info.labels['backend']}, while loading with {llm.__llm_backend__}."
      )
    return model
  except bentoml.exceptions.NotFound as err:
    if auto_import: return import_model(llm, trust_remote_code=llm.trust_remote_code)
    raise err from None

def load_model(llm: openllm.LLM[M, T], *decls: t.Any, **attrs: t.Any) -> M:
  '''Load the model from BentoML store.

  By default, it will try to find check the model in the local store.
  If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
  '''
  config, hub_attrs, attrs = process_config(llm.model_id, llm.trust_remote_code, **attrs)
  safe_serialization = openllm.utils.first_not_none(t.cast(
      t.Optional[bool], llm._bentomodel.info.metadata.get('safe_serialisation', None)),
                                                    attrs.pop('safe_serialization', None),
                                                    default=llm._serialisation_format == 'safetensors')
  if '_quantize' in llm._bentomodel.info.metadata and llm._bentomodel.info.metadata['_quantize'] == 'gptq':
    if not openllm.utils.is_autogptq_available():
      raise openllm.exceptions.OpenLLMException(
          "GPTQ quantisation requires 'auto-gptq' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\"'"
      )
    if llm.config['model_type'] != 'causal_lm':
      raise openllm.exceptions.OpenLLMException(
          f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
    return autogptq.AutoGPTQForCausalLM.from_quantized(llm._bentomodel.path,
                                                       *decls,
                                                       quantize_config=t.cast('autogptq.BaseQuantizeConfig',
                                                                              llm.quantization_config),
                                                       trust_remote_code=llm.trust_remote_code,
                                                       use_safetensors=safe_serialization,
                                                       **hub_attrs,
                                                       **attrs)

  device_map = attrs.pop('device_map', 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None)
  model = infer_autoclass_from_llm(llm, config).from_pretrained(llm._bentomodel.path,
                                                                *decls,
                                                                config=config,
                                                                trust_remote_code=llm.trust_remote_code,
                                                                device_map=device_map,
                                                                **hub_attrs,
                                                                **attrs).eval()
  if llm.__llm_backend__ in {'pt', 'vllm'}: check_unintialised_params(model)
  return t.cast('M', model)
