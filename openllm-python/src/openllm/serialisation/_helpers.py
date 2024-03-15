import contextlib, attr
from simple_di import Provide, inject
import bentoml, openllm
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.model import ModelOptions, ModelSignature
from openllm_core.exceptions import OpenLLMException
from openllm_core.utils import is_autogptq_available

_object_setattr = object.__setattr__


def get_hash(config) -> str:
  _commit_hash = getattr(config, '_commit_hash', None)
  if _commit_hash is None:
    raise ValueError(f'Cannot find commit hash in {config}')
  return _commit_hash


def patch_correct_tag(llm, config, _revision=None) -> None:
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
      _object_setattr(llm, '_tag', attr.evolve(llm.tag, version=_revision))  # HACK: This copies the correct revision into llm.tag
    if llm._revision is None:
      _object_setattr(llm, '_revision', _revision)  # HACK: This copies the correct revision into llm._model_version


def _create_metadata(llm, config, safe_serialisation, trust_remote_code, metadata=None):
  if metadata is None:
    metadata = {}
  metadata.update({'safe_serialisation': safe_serialisation, '_framework': llm.__llm_backend__})
  if llm.quantise:
    metadata['_quantize'] = llm.quantise
  architectures = getattr(config, 'architectures', [])
  if not architectures:
    if trust_remote_code:
      auto_map = getattr(config, 'auto_map', {})
      if not auto_map:
        raise RuntimeError(f'Failed to determine the architecture from both `auto_map` and `architectures` from {llm.model_id}')
      autoclass = 'AutoModelForSeq2SeqLM' if llm.config['model_type'] == 'seq2seq_lm' else 'AutoModelForCausalLM'
      if autoclass not in auto_map:
        raise RuntimeError(
          f"Given model '{llm.model_id}' is yet to be supported with 'auto_map'. OpenLLM currently only support encoder-decoders or decoders only models."
        )
      architectures = [auto_map[autoclass]]
    else:
      raise RuntimeError(
        'Failed to determine the architecture for this model. Make sure the `config.json` is valid and can be loaded with `transformers.AutoConfig`'
      )
  metadata.update({'_pretrained_class': architectures[0], '_revision': get_hash(config) if not llm.local else llm.revision})
  return metadata


def _create_signatures(llm, signatures=None):
  if signatures is None:
    signatures = {}
  if llm.__llm_backend__ == 'pt':
    if llm.quantise == 'gptq':
      if not is_autogptq_available():
        raise OpenLLMException("Requires 'auto-gptq' and 'optimum'. Install it with 'pip install \"openllm[gptq]\"'")
      signatures['generate'] = {'batchable': False}
    else:
      signatures.update({
        k: ModelSignature(batchable=False)
        for k in (
          '__call__',
          'forward',
          'generate',  #
          'contrastive_search',
          'greedy_search',  #
          'sample',
          'beam_search',
          'beam_sample',  #
          'group_beam_search',
          'constrained_beam_search',  #
        )
      })
  elif llm.__llm_backend__ == 'ctranslate':
    if llm.config['model_type'] == 'seq2seq_lm':
      non_batch_keys = {'score_file', 'translate_file'}
      batch_keys = {'generate_tokens', 'score_batch', 'translate_batch', 'translate_iterable', 'score_iterable'}
    else:
      non_batch_keys = set()
      batch_keys = {
        'async_generate_tokens',
        'forward_batch',
        'generate_batch',  #
        'generate_iterable',
        'generate_tokens',
        'score_batch',
        'score_iterable',  #
      }
    signatures.update({k: ModelSignature(batchable=False) for k in non_batch_keys})
    signatures.update({k: ModelSignature(batchable=True) for k in batch_keys})
  return signatures


@inject
@contextlib.contextmanager
def save_model(
  llm,
  config,
  safe_serialisation,  #
  trust_remote_code,
  module,
  external_modules,  #
  _model_store=Provide[BentoMLContainer.model_store],
  _api_version='v2.1.0',  #
):
  imported_modules = []
  bentomodel = bentoml.Model.create(
    llm.tag,
    module=f'openllm.serialisation.{module}',  #
    api_version=_api_version,
    options=ModelOptions(),  #
    context=openllm.utils.generate_context('openllm'),
    labels=openllm.utils.generate_labels(llm),
    metadata=_create_metadata(llm, config, safe_serialisation, trust_remote_code),
    signatures=_create_signatures(llm),
  )
  with openllm.utils.analytics.set_bentoml_tracking():
    try:
      bentomodel.enter_cloudpickle_context(external_modules, imported_modules)
      yield bentomodel, imported_modules
    except Exception:
      raise
    else:
      bentomodel.flush()
      bentomodel.save(_model_store)
      openllm.utils.analytics.track(
        openllm.utils.analytics.ModelSaveEvent(module=bentomodel.info.module, model_size_in_kb=openllm.utils.calc_dir_size(bentomodel.path) / 1024)
      )
    finally:
      bentomodel.exit_cloudpickle_context(imported_modules)
    return bentomodel
