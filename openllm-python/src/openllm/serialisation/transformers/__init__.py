import functools, importlib, logging, orjson, torch, transformers, openllm, attr
from huggingface_hub import snapshot_download
from openllm_core.exceptions import OpenLLMException
from openllm_core.utils import is_autogptq_available, is_flash_attn_2_available

from ._helpers import get_tokenizer, infer_autoclass_from_llm, process_config
from .weights import HfIgnore
from .._helpers import save_model, get_hash

logger = logging.getLogger(__name__)

__all__ = ['import_model', 'load_model']
_object_setattr = object.__setattr__

TOKENIZER_ATTRS = {'padding_side': 'left', 'truncation_side': 'left'}


@functools.lru_cache(maxsize=1)
def has_gpus() -> bool:
  try:
    from cuda import cuda

    err, *_ = cuda.cuInit(0)
    if err != cuda.CUresult.CUDA_SUCCESS:
      raise RuntimeError('Failed to initialise CUDA runtime binding.')
    err, _ = cuda.cuDeviceGetCount()
    if err != cuda.CUresult.CUDA_SUCCESS:
      raise RuntimeError('Failed to get CUDA device count.')
    return True
  except (ImportError, RuntimeError):
    return False


_TORCH_DTYPE_MAPPING = {
  'half': torch.float16,
  'float16': torch.float16,  #
  'float': torch.float32,
  'float32': torch.float32,  #
  'bfloat16': torch.bfloat16,
}


def _torch_dtype(dtype: str, model_id: str, trust_remote_code: bool) -> 'torch.dtype':
  hf_config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
  config_dtype = getattr(hf_config, 'torch_dtype', None)
  if config_dtype is None:
    config_dtype = torch.float32
  if dtype == 'auto':
    if config_dtype == torch.float32:
      torch_dtype = torch.float16
    else:
      torch_dtype = config_dtype
  else:
    torch_dtype = _TORCH_DTYPE_MAPPING.get(dtype, None)
    if torch_dtype is None:
      raise OpenLLMException(f'dtype not yet supported: {dtype}')
  if not torch.cuda.is_available() and torch_dtype != torch.float32:
    torch_dtype = torch.float32
  return torch_dtype


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
    'device_map': 'auto' if has_gpus() else None,
    'safe_serialization': _serialisation == 'safetensors',
    'torch_dtype': _torch_dtype(_dtype, _model_id, trust_remote_code),
  }
  attrs = {**_base_attrs, **attrs}
  config, hub_attrs, attrs = process_config(_model_id, trust_remote_code, **attrs)
  _revision = get_hash(config) if not _local else None
  if _revision:
    _bentomodel_tag = attr.evolve(_bentomodel_tag, version=_revision)
  model, tokenizer = (
    None,
    get_tokenizer(_model_id, trust_remote_code=trust_remote_code, **hub_attrs, **TOKENIZER_ATTRS),
  )
  with save_model(
    _bentomodel_tag,
    config,
    _serialisation,
    trust_remote_code,
    'transformers',
    [importlib.import_module(tokenizer.__module__)],
    _model_id,
    _quantize,
    _backend,
    _local,
    _dtype,
  ) as bentomodel:
    tokenizer.save_pretrained(bentomodel.path)
    if _quantization_config or (_quantize and _quantize not in {'squeezellm', 'awq'}):
      attrs['quantization_config'] = _quantization_config
    if _quantize == 'gptq' and _backend == 'pt':
      from optimum.gptq.constants import GPTQ_CONFIG

      with open(bentomodel.path_of(GPTQ_CONFIG), 'w', encoding='utf-8') as f:
        f.write(orjson.dumps(config.quantization_config, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS).decode())
    if _local:
      try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
          _model_id,
          *decls,
          local_files_only=True,
          config=config,
          trust_remote_code=trust_remote_code,
          **hub_attrs,
          **attrs,
        )
      except Exception:
        try:
          model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
            _model_id,
            *decls,
            local_files_only=True,
            config=config,
            trust_remote_code=trust_remote_code,
            **hub_attrs,
            **attrs,
          )
        except Exception as err:
          raise OpenLLMException(f'Failed to load model from {_model_id}') from err
      # for trust_remote_code to work
      bentomodel.enter_cloudpickle_context([importlib.import_module(model.__module__)], bentomodel.imported_modules)
      model.save_pretrained(bentomodel.path, max_shard_size='2GB', safe_serialization=_serialisation == 'safetensors')
      del model
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    else:
      # we will clone the all tings into the bentomodel path without loading model into memory
      snapshot_download(
        _model_id,
        local_dir=bentomodel.path,
        local_dir_use_symlinks=False,
        ignore_patterns=HfIgnore.ignore_patterns(_backend, _model_id),
      )
    return bentomodel


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
      model = auto_class.from_pretrained(
        llm.bentomodel.path, device_map=device_map, trust_remote_code=llm.trust_remote_code, **attrs
      )
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
        llm.bentomodel.path,
        *decls,
        config=config,
        trust_remote_code=llm.trust_remote_code,
        device_map=device_map,
        **attrs,
      )
  check_unintialised_params(model)

  # If OOM, then it is probably you don't have enough VRAM to run this model.
  loaded_in_kbit = (
    getattr(model, 'is_loaded_in_8bit', False)
    or getattr(model, 'is_loaded_in_4bit', False)
    or getattr(model, 'is_quantized', False)
  )
  if torch.cuda.is_available() and torch.cuda.device_count() == 1 and not loaded_in_kbit:
    try:
      model = model.to('cuda')
    except Exception as err:
      raise OpenLLMException(f'Failed to load model into GPU: {err}.\n') from err

  return model
