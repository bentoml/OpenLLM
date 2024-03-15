import importlib
import logging
import shutil

import transformers

import bentoml
from openllm_core.exceptions import OpenLLMException
from openllm_core.utils import is_ctranslate_available

from .._helpers import patch_correct_tag, save_model
from ..transformers._helpers import get_tokenizer, process_config

if not is_ctranslate_available():
  raise RuntimeError("'ctranslate2' is required to use with backend 'ctranslate'. Install it with 'pip install \"openllm[ctranslate]\"'")

import ctranslate2
from ctranslate2.converters.transformers import TransformersConverter

logger = logging.getLogger(__name__)


def _get_class(llm):
  return ctranslate2.Translator if llm.config['model_type'] == 'seq2seq_lm' else ctranslate2.Generator


def import_model(llm, *decls, trust_remote_code, **attrs):
  (_base_decls, _base_attrs), tokenizer_attrs = llm.llm_parameters
  for it in {'device_map', 'torch_dtype'}:
    _base_attrs.pop(it, None)  # pop out hf-specific attributes
  decls = (*_base_decls, *decls)
  attrs = {**_base_attrs, **attrs}
  low_cpu_mem_usage = attrs.pop('low_cpu_mem_usage', True)
  logger.debug(
    'Note that CTranslate2 will load into memory for conversion. Refer to https://opennmt.net/CTranslate2/guides/transformers.html for more information.'
  )
  if not llm._local:
    logger.warning(
      "It is RECOMMENDED to convert '%s' to CTranslate2 format yourself to utilise CTranslate2's features, then start with `openllm start /path/to/ct2-dir`. OpenLLM will conservely apply quantization for conversion if specified.",
      llm.model_id,
    )
  config, hub_attrs, attrs = process_config(llm.model_id, trust_remote_code, **attrs)
  patch_correct_tag(llm, config)
  tokenizer = get_tokenizer(llm.model_id, trust_remote_code=trust_remote_code, **hub_attrs, **tokenizer_attrs)
  with save_model(llm, config, False, trust_remote_code, 'ctranslate', [importlib.import_module(tokenizer.__module__)]) as save_metadata:
    bentomodel, _ = save_metadata
    if llm._local:
      shutil.copytree(
        llm.model_id, bentomodel.path, symlinks=False, ignore=shutil.ignore_patterns('.git', 'venv', '__pycache__', '.venv'), dirs_exist_ok=True
      )
    else:
      TransformersConverter(
        llm.model_id,
        load_as_float16=llm.quantise in ('float16', 'int8_float16'),
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code,
      ).convert(bentomodel.path, quantization=llm.quantise, force=True)
    # Save the original HF configuration to hf
    config.save_pretrained(bentomodel.path_of('/hf/'))
    tokenizer.save_pretrained(bentomodel.path)
    return bentomodel


def get(llm):
  try:
    model = bentoml.models.get(llm.tag)
    backend = model.info.labels['backend']
    if backend != llm.__llm_backend__:
      raise OpenLLMException(f"'{model.tag!s}' was saved with backend '{backend}', while loading with '{llm.__llm_backend__}'.")
    patch_correct_tag(
      llm,
      transformers.AutoConfig.from_pretrained(model.path_of('/hf/'), trust_remote_code=llm.trust_remote_code),
      _revision=model.info.metadata.get('_revision'),
    )
    return model
  except Exception as err:
    raise OpenLLMException(f'Failed while getting stored artefact (lookup for traceback):\n{err}') from err


def load_model(llm, *decls, **attrs):
  device = 'cuda' if llm._has_gpus else 'cpu'
  if llm.quantise:
    compute_type = llm.quantise
  elif llm.__llm_dtype__ == 'half':
    compute_type = 'float16'
  elif llm.__llm_dtype__ == 'float':
    compute_type = 'float32'
  else:
    compute_type = llm.__llm_dtype__
  return _get_class(llm)(llm.bentomodel.path, device=device, compute_type=compute_type)
