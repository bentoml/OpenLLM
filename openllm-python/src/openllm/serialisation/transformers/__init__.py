from __future__ import annotations
import importlib
import logging

import attr
import orjson
import torch
import transformers

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

from ._helpers import get_hash
from ._helpers import infer_autoclass_from_llm
from ._helpers import process_config
from .weights import HfIgnore


logger = logging.getLogger(__name__)

__all__ = ['import_model', 'get', 'load_model']
_object_setattr = object.__setattr__


def _patch_correct_tag(llm: openllm.LLM[M, T], config: transformers.PretrainedConfig, _revision: str | None = None):
  # NOTE: The following won't hit during local since we generated a correct version based on local path hash It will only hit if we use model from HF Hub
  if llm.revision is not None:
    return
  if not llm.local:
    try:
      if _revision is None:
        _revision = get_hash(config)
    except ValueError:
      pass
    if _revision is None and llm.tag.version is not None:
      _revision = llm.tag.version
    if llm.tag.version is None:
      _object_setattr(
        llm, '_tag', attr.evolve(llm.tag, version=_revision)
      )  # HACK: This copies the correct revision into llm.tag
    if llm._revision is None:
      _object_setattr(llm, '_revision', _revision)  # HACK: This copies the correct revision into llm._model_version


@inject
def import_model(llm, *decls, trust_remote_code, _model_store=Provide[BentoMLContainer.model_store], **attrs):
  _base_decls, _base_attrs = llm.llm_parameters[0]
  decls = (*_base_decls, *decls)
  attrs = {**_base_attrs, **attrs}
  if llm._local:
    logger.warning('Given model is a local model, OpenLLM will load model into memory for serialisation.')
  config, hub_attrs, attrs = process_config(llm.model_id, trust_remote_code, **attrs)
  _patch_correct_tag(llm, config)
  _, tokenizer_attrs = llm.llm_parameters
  quantize = llm.quantise
  safe_serialisation = openllm.utils.first_not_none(
    attrs.get('safe_serialization'), default=llm._serialisation == 'safetensors'
  )
  metadata = {'safe_serialisation': safe_serialisation}
  if quantize:
    metadata['_quantize'] = quantize
  architectures = getattr(config, 'architectures', [])
  if not architectures:
    raise RuntimeError(
      'Failed to determine the architecture for this model. Make sure the `config.json` is valid and can be loaded with `transformers.AutoConfig`'
    )
  metadata['_pretrained_class'] = architectures[0]
  if not llm._local:
    metadata['_revision'] = get_hash(config)
  else:
    metadata['_revision'] = llm.revision

  signatures = {}

  if quantize == 'gptq':
    if not openllm.utils.is_autogptq_available() or not openllm.utils.is_optimum_supports_gptq():
      raise openllm.exceptions.OpenLLMException(
        "GPTQ quantisation requires 'auto-gptq' and 'optimum' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\" --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/'"
      )
    signatures['generate'] = {'batchable': False}
  else:
    attrs['use_safetensors'] = safe_serialisation
    metadata['_framework'] = llm.__llm_backend__
    signatures.update(
      {
        k: ModelSignature(batchable=False)
        for k in (
          '__call__',
          'forward',
          'generate',
          'contrastive_search',
          'greedy_search',
          'sample',
          'beam_search',
          'beam_sample',
          'group_beam_search',
          'constrained_beam_search',
        )
      }
    )

  tokenizer = transformers.AutoTokenizer.from_pretrained(
    llm.model_id, trust_remote_code=trust_remote_code, **hub_attrs, **tokenizer_attrs
  )
  if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

  model = None
  external_modules = [importlib.import_module(tokenizer.__module__)]
  imported_modules = []
  bentomodel = bentoml.Model.create(
    llm.tag,
    module='openllm.serialisation.transformers',
    api_version='v2.1.0',
    options=ModelOptions(),
    context=openllm.utils.generate_context(framework_name='openllm'),
    labels=openllm.utils.generate_labels(llm),
    metadata=metadata,
    signatures=signatures,
  )
  with openllm.utils.analytics.set_bentoml_tracking():
    try:
      bentomodel.enter_cloudpickle_context(external_modules, imported_modules)
      tokenizer.save_pretrained(bentomodel.path)
      if llm._quantization_config or (llm.quantise and llm.quantise not in {'squeezellm', 'awq'}):
        attrs['quantization_config'] = llm.quantization_config
      if quantize == 'gptq':
        from optimum.gptq.constants import GPTQ_CONFIG

        with open(bentomodel.path_of(GPTQ_CONFIG), 'w', encoding='utf-8') as f:
          f.write(orjson.dumps(config.quantization_config, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode())
      if llm._local:  # possible local path
        model = infer_autoclass_from_llm(llm, config).from_pretrained(
          llm.model_id,
          *decls,
          local_files_only=True,
          config=config,
          trust_remote_code=trust_remote_code,
          **hub_attrs,
          **attrs,
        )
        # for trust_remote_code to work
        bentomodel.enter_cloudpickle_context([importlib.import_module(model.__module__)], imported_modules)
        model.save_pretrained(bentomodel.path, max_shard_size='2GB', safe_serialization=safe_serialisation)
        del model
        if torch.cuda.is_available():
          torch.cuda.empty_cache()
      else:
        # we will clone the all tings into the bentomodel path without loading model into memory
        snapshot_download(
          llm.model_id,
          local_dir=bentomodel.path,
          local_dir_use_symlinks=False,
          ignore_patterns=HfIgnore.ignore_patterns(llm),
        )
    except Exception:
      raise
    else:
      bentomodel.flush()  # type: ignore[no-untyped-call]
      bentomodel.save(_model_store)
      openllm.utils.analytics.track(
        openllm.utils.analytics.ModelSaveEvent(
          module=bentomodel.info.module, model_size_in_kb=openllm.utils.calc_dir_size(bentomodel.path) / 1024
        )
      )
    finally:
      bentomodel.exit_cloudpickle_context(imported_modules)
    return bentomodel


def get(llm):
  try:
    model = bentoml.models.get(llm.tag)
    backend = model.info.labels['backend']
    if backend != llm.__llm_backend__:
      raise openllm.exceptions.OpenLLMException(
        f"'{model.tag!s}' was saved with backend '{backend}', while loading with '{llm.__llm_backend__}'."
      )
    _patch_correct_tag(
      llm,
      transformers.AutoConfig.from_pretrained(model.path, trust_remote_code=llm.trust_remote_code),
      _revision=model.info.metadata.get('_revision'),
    )
    return model
  except Exception as err:
    raise openllm.exceptions.OpenLLMException(
      f'Failed while getting stored artefact (lookup for traceback):\n{err}'
    ) from err


def check_unintialised_params(model):
  unintialized = [n for n, param in model.named_parameters() if param.data.device == torch.device('meta')]
  if len(unintialized) > 0:
    raise RuntimeError(f'Found the following unintialized parameters in {model}: {unintialized}')


def load_model(llm, *decls, **attrs):
  if llm.quantise in {'awq', 'squeezellm'}:
    raise RuntimeError('AWQ is not yet supported with PyTorch backend.')
  config, attrs = transformers.AutoConfig.from_pretrained(
    llm.bentomodel.path, return_unused_kwargs=True, trust_remote_code=llm.trust_remote_code, **attrs
  )
  auto_class = infer_autoclass_from_llm(llm, config)
  device_map = attrs.pop('device_map', None)
  if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
      device_map = 'auto'
    elif torch.cuda.device_count() == 1:
      device_map = 'cuda:0'
  if llm.quantise in {'int8', 'int4'}:
    attrs['quantization_config'] = llm.quantization_config

  if '_quantize' in llm.bentomodel.info.metadata:
    _quantise = llm.bentomodel.info.metadata['_quantize']
    if _quantise == 'gptq':
      if not openllm.utils.is_autogptq_available() or not openllm.utils.is_optimum_supports_gptq():
        raise openllm.exceptions.OpenLLMException(
          "GPTQ quantisation requires 'auto-gptq' and 'optimum' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\" --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/'"
        )
      if llm.config['model_type'] != 'causal_lm':
        raise openllm.exceptions.OpenLLMException(
          f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})"
        )

    # TODO: investigate load with flash attention
    model = auto_class.from_pretrained(
      llm.bentomodel.path, device_map=device_map, trust_remote_code=llm.trust_remote_code, **attrs
    )
  else:
    model = auto_class.from_pretrained(
      llm.bentomodel.path,
      *decls,
      config=config,
      trust_remote_code=llm.trust_remote_code,
      device_map=device_map,
      **attrs,
    )
  check_unintialised_params(model)
  return model
