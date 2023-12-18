from __future__ import annotations
import importlib, logging
import orjson, torch, transformers, bentoml, openllm

from huggingface_hub import snapshot_download
from openllm_core.exceptions import OpenLLMException
from openllm_core.utils import first_not_none, is_autogptq_available, is_flash_attn_2_available

from ._helpers import get_tokenizer, infer_autoclass_from_llm, process_config
from .weights import HfIgnore
from .._helpers import patch_correct_tag, save_model

logger = logging.getLogger(__name__)

__all__ = ['import_model', 'get', 'load_model']
_object_setattr = object.__setattr__


def import_model(llm, *decls, trust_remote_code, **attrs):
  (_base_decls, _base_attrs), tokenizer_attrs = llm.llm_parameters
  decls = (*_base_decls, *decls)
  attrs = {**_base_attrs, **attrs}
  if llm._local:
    logger.warning('Given model is a local model, OpenLLM will load model into memory for serialisation.')
  config, hub_attrs, attrs = process_config(llm.model_id, trust_remote_code, **attrs)
  patch_correct_tag(llm, config)
  safe_serialisation = first_not_none(attrs.get('safe_serialization'), default=llm._serialisation == 'safetensors')
  if llm.quantise != 'gptq':
    attrs['use_safetensors'] = safe_serialisation

  model = None
  tokenizer = get_tokenizer(llm.model_id, trust_remote_code=trust_remote_code, **hub_attrs, **tokenizer_attrs)
  with save_model(
    llm, config, safe_serialisation, trust_remote_code, 'transformers', [importlib.import_module(tokenizer.__module__)]
  ) as save_metadata:
    bentomodel, imported_modules = save_metadata
    tokenizer.save_pretrained(bentomodel.path)
    if llm._quantization_config or (llm.quantise and llm.quantise not in {'squeezellm', 'awq'}):
      attrs['quantization_config'] = llm.quantization_config
    if llm.quantise == 'gptq' and llm.__llm_backend__ == 'pt':
      from optimum.gptq.constants import GPTQ_CONFIG

      with open(bentomodel.path_of(GPTQ_CONFIG), 'w', encoding='utf-8') as f:
        f.write(orjson.dumps(config.quantization_config, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode())
    if llm._local:  # possible local path
      model = infer_autoclass_from_llm(llm, config).from_pretrained(
        llm.model_id, *decls, local_files_only=True, config=config, trust_remote_code=trust_remote_code, **hub_attrs, **attrs
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
        local_dir=bentomodel.path,  #
        local_dir_use_symlinks=False,
        ignore_patterns=HfIgnore.ignore_patterns(llm),  #
      )
    return bentomodel


def get(llm):
  try:
    model = bentoml.models.get(llm.tag)
    backend = model.info.labels['backend']
    if backend != llm.__llm_backend__:
      raise OpenLLMException(f"'{model.tag!s}' was saved with backend '{backend}', while loading with '{llm.__llm_backend__}'.")
    patch_correct_tag(
      llm,
      transformers.AutoConfig.from_pretrained(model.path, trust_remote_code=llm.trust_remote_code),
      _revision=model.info.metadata.get('_revision'),
    )
    return model
  except Exception as err:
    raise OpenLLMException(f'Failed while getting stored artefact (lookup for traceback):\n{err}') from err


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
  if llm.__llm_backend__ == 'triton':
    return openllm.models.load_model(llm, config, **attrs)

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
    if _quantise == 'gptq' and llm.__llm_backend__ == 'pt':
      if not is_autogptq_available():
        raise OpenLLMException(
          "GPTQ quantisation requires 'auto-gptq' and 'optimum' (Not found in local environment). Install it with 'pip install \"openllm[gptq]\" --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/'"
        )
      if llm.config['model_type'] != 'causal_lm':
        raise OpenLLMException(f"GPTQ only support Causal LM (got {llm.__class__} of {llm.config['model_type']})")

    try:
      model = auto_class.from_pretrained(
        llm.bentomodel.path,
        device_map=device_map,
        trust_remote_code=llm.trust_remote_code,
        use_flash_attention_2=is_flash_attn_2_available(),
        **attrs,
      )
    except Exception as err:
      logger.debug("Failed to load model with 'use_flash_attention_2' (lookup for traceback):\n%s", err)
      model = auto_class.from_pretrained(llm.bentomodel.path, device_map=device_map, trust_remote_code=llm.trust_remote_code, **attrs)
  else:
    try:
      model = auto_class.from_pretrained(
        llm.bentomodel.path,
        *decls,
        config=config,
        trust_remote_code=llm.trust_remote_code,
        device_map=device_map,
        use_flash_attention_2=is_flash_attn_2_available(),
        **attrs,
      )
    except Exception as err:
      logger.debug("Failed to load model with 'use_flash_attention_2' (lookup for traceback):\n%s", err)
      model = auto_class.from_pretrained(
        llm.bentomodel.path, *decls, config=config, trust_remote_code=llm.trust_remote_code, device_map=device_map, **attrs
      )
  check_unintialised_params(model)
  return model
