'''Serialisation related implementation for Transformers-based implementation.'''
from __future__ import annotations
import importlib
import logging
import typing as t

import attr
import orjson

from packaging.version import Version
from huggingface_hub import snapshot_download
from simple_di import Provide
from simple_di import inject

import bentoml
import openllm

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.model import ModelOptions
from bentoml._internal.models.model import ModelSignature
from openllm_core._typing_compat import M
from openllm_core._typing_compat import T

from ._helpers import check_unintialised_params
from ._helpers import get_hash
from ._helpers import infer_autoclass_from_llm
from ._helpers import process_config
from .weights import HfIgnore

if t.TYPE_CHECKING:
  import types

  import auto_gptq as autogptq
  import torch
  import torch.nn
  import transformers

  from bentoml._internal.models import ModelStore
  from openllm_core._typing_compat import DictStrAny
else:
  transformers = openllm.utils.LazyLoader('transformers', globals(), 'transformers')
  autogptq = openllm.utils.LazyLoader('autogptq', globals(), 'auto_gptq')
  torch = openllm.utils.LazyLoader('torch', globals(), 'torch')

logger = logging.getLogger(__name__)

__all__ = ['import_model', 'get', 'load_model']
_object_setattr = object.__setattr__

def _patch_correct_tag(llm: openllm.LLM[M, T], config: transformers.PretrainedConfig, _revision: str | None = None) -> None:
  # NOTE: The following won't hit during local since we generated a correct version based on local path hash It will only hit if we use model from HF Hub
  if not llm._local:
    if _revision is None: _revision = get_hash(config)
    if llm._revision is None: _object_setattr(llm, '_revision', _revision)  # HACK: This copies the correct revision into llm._model_version
    if llm._tag.version is None: _object_setattr(llm, '_tag', attr.evolve(llm.tag, version=_revision))  # HACK: This copies the correct revision into llm.tag

@inject
def import_model(llm: openllm.LLM[M, T], *decls: t.Any, trust_remote_code: bool, _model_store: ModelStore = Provide[BentoMLContainer.model_store], **attrs: t.Any) -> bentoml.Model:
  """Auto detect model type from given model_id and import it to bentoml's model store.

  For all kwargs, it will be parsed into `transformers.AutoConfig.from_pretrained` first,
  returning all of the unused kwargs.
  The unused kwargs then parsed directly into AutoModelForSeq2SeqLM or AutoModelForCausalLM.
  For all tokenizer kwargs, make sure to prefix it with `_tokenizer_` to avoid confusion.

  Note: Currently, there are only two tasks supported: `text-generation` and `text2text-generation`.

  Refer to Transformers documentation for more information about kwargs.

  Args:
    llm: The LLM instance for this given model.
    trust_remote_code: Whether to trust the remote code when loading the model.
    *decls: Args to be passed into AutoModelForSeq2SeqLM or AutoModelForCausalLM.
    **attrs: Kwargs to be passed into AutoModelForSeq2SeqLM or AutoModelForCausalLM.
  """
  config, hub_attrs, attrs = process_config(llm.model_id, trust_remote_code, **attrs)
  _patch_correct_tag(llm, config)
  _, tokenizer_attrs = llm.llm_parameters
  quantize = llm._quantise
  safe_serialisation = openllm.utils.first_not_none(attrs.get('safe_serialization'), default=llm._serialisation == 'safetensors')
  metadata: DictStrAny = {'safe_serialisation': safe_serialisation}
  if quantize: metadata['_quantize'] = quantize
  architectures = getattr(config, 'architectures', [])
  if not architectures: raise RuntimeError('Failed to determine the architecture for this model. Make sure the `config.json` is valid and can be loaded with `transformers.AutoConfig`')
  metadata['_pretrained_class'] = architectures[0]
  metadata['_revision'] = get_hash(config)

  signatures: DictStrAny = {}

  if quantize == 'gptq':
    if not openllm.utils.is_autogptq_available() or not openllm.utils.is_optimum_supports_gptq():
      raise openllm.exceptions.OpenLLMException(
          "GPTQ quantisation requires 'auto-gptq' and 'optimum' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\" --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/'"
      )
    signatures['generate'] = {'batchable': False}
  else:
    attrs['use_safetensors'] = safe_serialisation
    metadata['_framework'] = llm.__llm_backend__
    signatures.update({
        k: ModelSignature(batchable=False)
        for k in ('__call__', 'forward', 'generate', 'contrastive_search', 'greedy_search', 'sample', 'beam_search', 'beam_sample', 'group_beam_search', 'constrained_beam_search')
    })

  tokenizer = transformers.AutoTokenizer.from_pretrained(llm.model_id, trust_remote_code=trust_remote_code, **hub_attrs, **tokenizer_attrs)
  if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

  model = None
  external_modules: list[types.ModuleType] = [importlib.import_module(tokenizer.__module__)]
  imported_modules: list[types.ModuleType] = []
  bentomodel = bentoml.Model.create(llm.tag,
                                    module='openllm.serialisation.transformers',
                                    api_version='v2.1.0',
                                    options=ModelOptions(),
                                    context=openllm.utils.generate_context(framework_name='openllm'),
                                    labels=openllm.utils.generate_labels(llm),
                                    metadata=metadata,
                                    signatures=signatures)
  with openllm.utils.analytics.set_bentoml_tracking():
    try:
      bentomodel.enter_cloudpickle_context(external_modules, imported_modules)
      tokenizer.save_pretrained(bentomodel.path)
      if llm._quantise or llm._quantization_config: attrs['quantization_config'] = llm.quantization_config
      if quantize == 'gptq':
        from optimum.gptq.constants import GPTQ_CONFIG
        with open(bentomodel.path_of(GPTQ_CONFIG), 'w', encoding='utf-8') as f:
          f.write(orjson.dumps(config.quantization_config, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode())
      if llm._local:  # possible local path
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
      del model
    return bentomodel

def get(llm: openllm.LLM[M, T], auto_import: bool = False) -> bentoml.Model:
  '''Return an instance of ``bentoml.Model`` from given LLM instance.

  By default, it will try to check the model in the local store.
  If model is not found, and ``auto_import`` is set to True, it will try to import the model from HuggingFace Hub.

  Otherwise, it will raises a ``bentoml.exceptions.NotFound``.
  '''
  try:
    model = bentoml.models.get(llm.tag)
    backend = model.info.labels['backend']
    if Version(model.info.api_version) < Version('v2.1.0'): raise openllm.exceptions.OpenLLMException("Please run 'openllm prune -y --include-bentos' (model saved <2.1.0).")
    if backend != llm.__llm_backend__: raise openllm.exceptions.OpenLLMException(f"'{model.tag!s}' was saved with backend '{backend}', while loading with '{llm.__llm_backend__}'.")
    _patch_correct_tag(llm, process_config(model.path, llm.trust_remote_code)[0], _revision=model.info.metadata['_revision'])
    return model
  except Exception as err:
    if auto_import: return import_model(llm, trust_remote_code=llm.trust_remote_code)
    raise openllm.exceptions.OpenLLMException(f'Failed while getting stored artefact (lookup for traceback):\n{err}') from err

def load_model(llm: openllm.LLM[M, T], *decls: t.Any, **attrs: t.Any) -> M:
  config, hub_attrs, attrs = process_config(llm.bentomodel.path, llm.trust_remote_code, **attrs)
  _patch_correct_tag(llm, config, _revision=llm.bentomodel.info.metadata['_revision'])
  auto_class = infer_autoclass_from_llm(llm, config)
  device_map: str | None = attrs.pop('device_map', 'auto' if torch.cuda.is_available() and torch.cuda.device_count() > 1 else None)
  if llm._quantise or llm._quantization_config: attrs['quantization_config'] = llm.quantization_config

  if '_quantize' in llm.bentomodel.info.metadata and llm.bentomodel.info.metadata['_quantize'] == 'gptq':
    if not openllm.utils.is_autogptq_available() or not openllm.utils.is_optimum_supports_gptq():
      raise openllm.exceptions.OpenLLMException(
          "GPTQ quantisation requires 'auto-gptq' and 'optimum' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\" --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/'"
      )
    if llm.config['model_type'] != 'causal_lm': raise openllm.exceptions.OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")

    try:
      model = auto_class.from_pretrained(llm.bentomodel.path, device_map='auto', use_flash_attention_2=True, **hub_attrs, **attrs)
    except Exception as err:
      logger.debug("Exception caught while trying to load with 'flash_attention_2': %s", err)
      model = auto_class.from_pretrained(llm.bentomodel.path, device_map='auto', use_flash_attention_2=False, **hub_attrs, **attrs)
    # XXX: Use the below logic once TheBloke finished migration to new GPTQConfig from transformers
    # Seems like the logic below requires to add support for safetensors on accelerate
    # from accelerate import init_empty_weights
    # from optimum.gptq import load_quantized_model
    # # disable exllama if gptq is loaded on CPU
    # disable_exllama = not torch.cuda.is_available()
    # with init_empty_weights():
    #   empty = auto_class.from_pretrained(llm.model_id, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, device_map='auto')
    # empty.tie_weights()
    # model = load_quantized_model(empty, save_folder=llm._bentomodel.path, device_map='auto', disable_exllama=disable_exllama)
  else:
    model = auto_class.from_pretrained(llm.bentomodel.path, *decls, config=config, trust_remote_code=llm.trust_remote_code, device_map=device_map, **hub_attrs, **attrs).eval()
    if llm.__llm_backend__ == 'pt': check_unintialised_params(model)
  return t.cast('M', model)
