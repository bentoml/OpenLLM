'''Serialisation related implementation for Transformers-based implementation.'''
from __future__ import annotations
import importlib
import logging
import typing as t

from huggingface_hub import snapshot_download
from packaging.version import Version
from simple_di import Provide
from simple_di import inject

import bentoml
import openllm

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.model import ModelOptions
from openllm_core._typing_compat import M
from openllm_core._typing_compat import T

from ._helpers import check_unintialised_params
from ._helpers import infer_autoclass_from_llm
from ._helpers import infer_tokenizers_from_llm
from ._helpers import make_model_signatures
from ._helpers import process_config
from .weights import HfIgnore

if t.TYPE_CHECKING:
  import types

  import auto_gptq as autogptq
  import torch
  import torch.nn

  from bentoml._internal.models import ModelStore
  from openllm_core._typing_compat import DictStrAny
else:
  autogptq = openllm.utils.LazyLoader('autogptq', globals(), 'auto_gptq')
  torch = openllm.utils.LazyLoader('torch', globals(), 'torch')

logger = logging.getLogger(__name__)

__all__ = ['import_model', 'get', 'load_model']

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
  quantize = llm._quantize
  safe_serialisation = openllm.utils.first_not_none(attrs.get('safe_serialization'), default=llm._serialisation == 'safetensors')
  # Disable safe serialization with vLLM
  if llm.__llm_backend__ == 'vllm': safe_serialisation = False
  metadata: DictStrAny = {'safe_serialisation': safe_serialisation}
  if quantize: metadata['_quantize'] = quantize
  architectures = getattr(config, 'architectures', [])
  if not architectures: raise RuntimeError('Failed to determine the architecture for this model. Make sure the `config.json` is valid and can be loaded with `transformers.AutoConfig`')
  metadata['_pretrained_class'] = architectures[0]

  signatures: DictStrAny = {}

  if quantize == 'gptq':
    if not openllm.utils.is_autogptq_available() or not openllm.utils.is_optimum_supports_gptq():
      raise openllm.exceptions.OpenLLMException(
          "GPTQ quantisation requires 'auto-gptq' and 'optimum' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\" --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/'"
      )
    if llm.config['model_type'] != 'causal_lm':
      raise openllm.exceptions.OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")
    signatures['generate'] = {'batchable': False}
  else:
    # this model might be called with --quantize int4, therefore we need to pop this out
    # since saving int4 is not yet supported
    if 'quantization_config' in attrs and getattr(attrs['quantization_config'], 'load_in_4bit', False):
      attrs.pop('quantization_config')
    if llm.__llm_backend__ != 'flax': attrs['use_safetensors'] = safe_serialisation
    metadata['_framework'] = llm.__llm_backend__
    signatures.update(make_model_signatures(llm))

  tokenizer = infer_tokenizers_from_llm(llm).from_pretrained(llm.model_id, trust_remote_code=trust_remote_code, **hub_attrs, **tokenizer_attrs)
  if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

  external_modules: list[types.ModuleType] = [importlib.import_module(tokenizer.__module__)]
  imported_modules: list[types.ModuleType] = []
  bentomodel = bentoml.Model.create(llm.tag,
                                    module='openllm.serialisation.transformers',
                                    api_version='v2',
                                    options=ModelOptions(),
                                    context=openllm.utils.generate_context(framework_name='openllm'),
                                    labels=openllm.utils.generate_labels(llm),
                                    metadata=metadata,
                                    signatures=signatures)
  with openllm.utils.analytics.set_bentoml_tracking():
    try:
      bentomodel.enter_cloudpickle_context(external_modules, imported_modules)
      tokenizer.save_pretrained(bentomodel.path)
      if llm._local:
        # possible local path
        logger.debug('Model will be loaded into memory to save to target store as it is from local path.')
        model = infer_autoclass_from_llm(llm, config).from_pretrained(llm.model_id, *decls, config=config, trust_remote_code=trust_remote_code, **hub_attrs, **attrs)
        # for trust_remote_code to work
        bentomodel.enter_cloudpickle_context([importlib.import_module(model.__module__)], imported_modules)
        model.save_pretrained(bentomodel.path, max_shard_size='5GB', safe_serialization=safe_serialisation)
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
  '''Return an instance of ``bentoml.Model`` from given LLM instance.

  By default, it will try to check the model in the local store.
  If model is not found, and ``auto_import`` is set to True, it will try to import the model from HuggingFace Hub.

  Otherwise, it will raises a ``bentoml.exceptions.NotFound``.
  '''
  try:
    model = bentoml.models.get(llm.tag)
    if Version(model.info.api_version) < Version('v2'):
      raise openllm.exceptions.OpenLLMException('Please run "openllm prune -y --include-bentos" and upgrade all saved model to latest release.')
    if model.info.labels['backend'] != llm.__llm_backend__:
      raise openllm.exceptions.OpenLLMException(f"Model {model.tag} was saved with backend {model.info.labels['backend']}, while loading with {llm.__llm_backend__}.")
    return model
  except Exception as err:
    if auto_import: return import_model(llm, trust_remote_code=llm.trust_remote_code)
    raise openllm.exceptions.OpenLLMException(f'Failed while getting stored artefact (lookup for traceback):\n{err}') from err

def load_model(llm: openllm.LLM[M, T], *decls: t.Any, **attrs: t.Any) -> M:
  config, hub_attrs, attrs = process_config(llm.model_id, llm.trust_remote_code, **attrs)
  auto_class = infer_autoclass_from_llm(llm, config)
  device_map: str | None = attrs.pop('device_map', 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None)

  if '_quantize' in llm._bentomodel.info.metadata and llm._bentomodel.info.metadata['_quantize'] == 'gptq':
    if not openllm.utils.is_autogptq_available() or not openllm.utils.is_optimum_supports_gptq():
      raise openllm.exceptions.OpenLLMException(
          "GPTQ quantisation requires 'auto-gptq' and 'optimum' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\" --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/'"
      )
    if llm.config['model_type'] != 'causal_lm': raise openllm.exceptions.OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")

    model = auto_class.from_pretrained(llm._bentomodel.path, device_map='auto', **hub_attrs, **attrs)
    # TODO: Use the below logic once TheBloke finished migration to new GPTQConfig from transformers
    # from accelerate import init_empty_weights
    # from optimum.gptq import load_quantized_model
    # # disable exllama if gptq is loaded on CPU
    # disable_exllama = not torch.cuda.is_available()
    # with init_empty_weights():
    #   empty = auto_class.from_pretrained(llm.model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map='auto')
    # empty.tie_weights()
    # model = load_quantized_model(empty, save_folder=llm._bentomodel.path, device_map='auto', disable_exllama=disable_exllama)
  else:
    model = auto_class.from_pretrained(llm._bentomodel.path, *decls, config=config, trust_remote_code=llm.trust_remote_code, device_map=device_map, **hub_attrs, **attrs).eval()
    if llm.__llm_backend__ in {'pt', 'vllm'}: check_unintialised_params(model)
  return t.cast('M', model)
