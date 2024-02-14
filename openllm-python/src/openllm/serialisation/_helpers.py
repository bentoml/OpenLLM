from __future__ import annotations
import contextlib, attr, bentoml, openllm, types, logging, typing as t
from simple_di import Provide, inject
from bentoml._internal.configuration.containers import BentoMLContainer
from openllm_core._typing_compat import LiteralSerialisation, LiteralQuantise, LiteralBackend

if t.TYPE_CHECKING:
  import transformers
  from bentoml._internal.models import ModelStore

_object_setattr = object.__setattr__
logger = logging.getLogger(__name__)


def get_hash(config: transformers.PretrainedConfig) -> str:
  _commit_hash = getattr(config, '_commit_hash', None)
  if _commit_hash is None:
    logger.warning('Cannot find commit hash in %r', config)
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
      _object_setattr(
        llm, '_tag', attr.evolve(llm.tag, version=_revision)
      )  # HACK: This copies the correct revision into llm.tag
    if llm._revision is None:
      _object_setattr(llm, '_revision', _revision)  # HACK: This copies the correct revision into llm._model_version


def _create_metadata(llm, config, safe_serialisation, trust_remote_code, metadata=None):
  if metadata is None:
    metadata = {}
  metadata.update({'_framework': llm.__llm_backend__})
  if llm.quantise:
    metadata['_quantize'] = llm.quantise
  architectures = getattr(config, 'architectures', [])
  if not architectures:
    if trust_remote_code:
      auto_map = getattr(config, 'auto_map', {})
      if not auto_map:
        raise RuntimeError(
          f'Failed to determine the architecture from both `auto_map` and `architectures` from {llm.model_id}'
        )
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
  metadata.update({
    '_pretrained_class': architectures[0],
    '_revision': get_hash(config) if not llm.local else llm.revision,
    '_local': llm.local,
    'serialisation': llm._serialisation,
    'model_name': llm.config['model_name'],
    'architecture': llm.config['architecture'],
    'model_id': llm.model_id,
  })
  return metadata


@attr.define(init=False)
class _Model(bentoml.Model):
  _imported_modules: t.List[types.ModuleType] = None

  @property
  def imported_modules(self):
    if self._imported_modules is None:
      self._imported_modules = []
    return self._imported_modules

  @imported_modules.setter
  def imported_modules(self, value):
    self._imported_modules = value

  @classmethod
  def create(cls, tag, *, module, api_version, labels=None, metadata=None):
    return super().create(
      tag,
      module=module,
      api_version=api_version,
      signatures={},
      labels=labels,
      metadata=metadata,
      context=openllm.utils.generate_context('openllm'),
    )


@inject
@contextlib.contextmanager
def save_model(
  tag: bentoml.Tag,
  config: transformers.PretrainedConfig,
  serialisation: LiteralSerialisation,  #
  trust_remote_code: bool,
  module: str,
  external_modules: list[types.ModuleType],  #
  model_id: str,
  quantise: LiteralQuantise,
  backend: LiteralBackend,
  _local: bool,
  _dtype: str,
  _model_store: ModelStore = Provide[BentoMLContainer.model_store],
  _api_version: str = 'v3.0.0',  #
) -> bentoml.Model:
  imported_modules = []
  architectures = getattr(config, 'architectures', [])
  _metadata = {
    'model_id': model_id,
    'backend': backend,
    'dtype': _dtype,
    'architectures': architectures,
    '_revision': get_hash(config) or tag.version,
    '_local': _local,
    'serialisation': serialisation,
  }
  if quantise:
    _metadata['_quantize'] = quantise
  bentomodel = _Model.create(
    tag,
    module=f'openllm.serialisation.{module}',
    api_version=_api_version,
    labels=openllm.utils.generate_labels(serialisation),
    metadata=_metadata,
  )
  with openllm.utils.analytics.set_bentoml_tracking():
    try:
      bentomodel.enter_cloudpickle_context(external_modules, imported_modules)
      bentomodel.imported_modules = imported_modules
      yield bentomodel
    except Exception:
      raise
    else:
      bentomodel.flush()
      bentomodel.save(_model_store)
      openllm.utils.analytics.track(
        openllm.utils.analytics.ModelSaveEvent(
          module=bentomodel.info.module, model_size_in_kb=openllm.utils.calc_dir_size(bentomodel.path) / 1024
        )
      )
    finally:
      bentomodel.exit_cloudpickle_context(bentomodel.imported_modules)
    return bentomodel
