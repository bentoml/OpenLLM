from __future__ import annotations

import logging, os, attr, inflection, openllm_core, importlib.metadata, typing as t
from openllm_core.utils import LazyLoader, getenv, resolve_filepath, validate_is_path, normalise_model_name
from openllm.serialisation import _make_tag_components
from openllm.exceptions import OpenLLMException

logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
  import bentoml, transformers, torch
else:
  transformers = LazyLoader('transformers', globals(), 'transformers')
  bentoml = LazyLoader('bentoml', globals(), 'bentoml')
  torch = LazyLoader('torch', globals(), 'torch')


def prepare_model(
  model_id: str,
  /,
  *decls: t.Any,
  bentomodel_tag=None,
  bentomodel_version=None,
  quantize=None,
  dtype='auto',
  serialisation='safetensors',
  trust_remote_code=False,
  low_cpu_mem_usage=True,
  **kwargs,
):
  try:
    # NOTE in case model_id is a custom bentomodel tag
    return bentoml.models.get(model_id.lower())
  except Exception:
    logger.debug('Failed to load model from %s', model_id, exc_info=True)
    pass

  _local = False
  if validate_is_path(model_id):
    model_id, _local = resolve_filepath(model_id), True
  dtype = getenv('dtype', default=dtype, var=['TORCH_DTYPE'])
  if dtype is None:
    logger.warning('Setting dtype to auto. Inferring from %s %s.', 'remote' if not _local else 'local', model_id)
    dtype = 'auto'
  quantize = getenv('quantize', default=quantize, var=['QUANITSE'])

  kwargs.update({'low_cpu_mem_usage': low_cpu_mem_usage})

  if bentomodel_tag is None:
    model_tag, model_version = _make_tag_components(model_id, bentomodel_version)
    if bentomodel_version is not None:
      model_version = bentomodel_version
    bentomodel_tag = bentoml.Tag.from_taglike(f'{model_tag}:{model_version}' if model_version else model_tag)
  else:
    bentomodel_tag = bentoml.Tag.from_taglike(bentomodel_tag)

  try:
    return bentoml.models.get(bentomodel_tag)
  except bentoml.exceptions.NotFound:
    config = transformers.AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code, **kwargs)

    _revision = None
    if not _local:
      _revision = getattr(config, '_commit_hash', None)
      if _revision is None:
        logger.warning('Cannot find commit hash in %r', config)
      else:
        bentomodel_tag = attr.evolve(bentomodel_tag, version=_revision)

    metadata = {
      'model_id': model_id,
      'dtype': dtype,
      '_revision': _revision or bentomodel_tag.version,
      '_local': _local,
      'serialisation': serialisation,
      'architectures': config.architectures,
      'trust_remote_code': trust_remote_code,
      'api_version': '0.5.0',
      'llm_type': normalise_model_name(model_id),
      **{
        f'{inflection.underscore(package)}_version': importlib.metadata.version(package)
        for package in {'openllm', 'openllm-core', 'openllm-client'}
      },
    }
    labels = {}
    if quantize:
      metadata['quantize'] = quantize
      labels['quantization'] = quantize
    model, tokenizer = None, None

    module = f'{os.path.basename(os.path.dirname(__file__))}.{os.path.basename(__file__)[:-3]}'
    with bentoml.models.create(bentomodel_tag, labels=labels, metadata=metadata, module=module) as bentomodel:
      arch = config.architectures[0]
      if arch not in openllm_core.AutoConfig._architecture_mappings:
        raise OpenLLMException(f'Failed to determine config class for {model_id}')
      llm_config: openllm_core.LLMConfig = openllm_core.AutoConfig.for_model(
        openllm_core.AutoConfig._architecture_mappings[arch]
      ).model_construct_env()

      with open(bentomodel.path_of('openllm_configuration.json'), 'w') as f:
        f.write(llm_config.model_dump_json())

      if _local:
        logger.warning('Loading local model %s into memory', model_id)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code, **kwargs)
        try:
          model = transformers.AutoModelForCasualLM.from_pretrained(
            model_id, *decls, local_files_only=True, config=config, trust_remote_code=trust_remote_code, **kwargs
          )
        except Exception:
          try:
            model = transformers.AutoModelForSeq2SeqLM.from_pretrained(
              model_id, *decls, local_files_only=True, config=config, trust_remote_code=trust_remote_code, **kwargs
            )
          except Exception as err:
            raise OpenLLMException(f'Failed to load model from {model_id}') from err
        model.save_pretrained(bentomodel.path, safe_serialization=serialisation == 'safetensors')
        tokenizer.save_pretrained(bentomodel.path)

        del model, tokenizer
        if torch.cuda.is_available():
          torch.cuda.empty_cache()

      return bentomodel
