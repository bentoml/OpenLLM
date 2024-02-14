import importlib, logging, shutil
from openllm_core.utils import is_ctranslate_available

from .._helpers import save_model
from ..transformers._helpers import get_tokenizer, process_config
from ..transformers import _torch_dtype, TOKENIZER_ATTRS, _has_gpus

if not is_ctranslate_available():
  raise RuntimeError(
    "'ctranslate2' is required to use with backend 'ctranslate'. Install it with 'pip install \"openllm[ctranslate]\"'"
  )
import ctranslate2
from ctranslate2.converters.transformers import TransformersConverter

logger = logging.getLogger(__name__)


def _get_class(llm):
  return ctranslate2.Translator if llm.config['model_type'] == 'seq2seq_lm' else ctranslate2.Generator


def import_model(
  *decls,
  _model_id=None,
  _bentomodel_tag=None,
  _backend=None,
  _local=False,
  _quantization_config=None,
  _quantize=None,
  _dtype='auto',
  _serialisation='safetensors',
  trust_remote_code,
  **attrs,
):
  _base_attrs = {
    'device_map': 'auto' if _has_gpus() else None,
    'safe_serialization': _serialisation == 'safetensors',
    'torch_dtype': _torch_dtype(_dtype, _model_id, trust_remote_code),
  }
  attrs = {**_base_attrs, **attrs}
  for it in {'device_map', 'torch_dtype'}:
    _base_attrs.pop(it, None)  # pop out hf-specific attributes
  low_cpu_mem_usage = attrs.pop('low_cpu_mem_usage', True)
  logger.debug(
    'Note that CTranslate2 will load into memory for conversion. Refer to https://opennmt.net/CTranslate2/guides/transformers.html for more information.'
  )
  if not _local:
    logger.warning(
      "It is RECOMMENDED to convert '%s' to CTranslate2 format yourself to utilise CTranslate2's features, then start with `openllm start /path/to/ct2-dir`. OpenLLM will conservely apply quantization for conversion if specified.",
      _model_id,
    )
  config, hub_attrs, attrs = process_config(_model_id, trust_remote_code, **attrs)
  tokenizer = get_tokenizer(_model_id, trust_remote_code=trust_remote_code, **hub_attrs, **TOKENIZER_ATTRS)
  with save_model(
    _bentomodel_tag,
    config,
    'ctranslate',
    trust_remote_code,
    'ctranslate',
    [importlib.import_module(tokenizer.__module__)],
    _model_id,
    _quantize,
    _backend,
    _local,
  ) as bentomodel:
    if _local:
      shutil.copytree(
        _model_id,
        bentomodel.path,
        symlinks=False,
        ignore=shutil.ignore_patterns('.git', 'venv', '__pycache__', '.venv'),
        dirs_exist_ok=True,
      )
    else:
      TransformersConverter(
        _model_id,
        load_as_float16=_quantize in ('float16', 'int8_float16'),
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code,
      ).convert(bentomodel.path, quantization=_quantize, force=True)
    # Save the original HF configuration to hf
    config.save_pretrained(bentomodel.path_of('/hf/'))
    tokenizer.save_pretrained(bentomodel.path)
    return bentomodel


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
