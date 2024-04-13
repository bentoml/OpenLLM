from __future__ import annotations
import importlib, logging, inspect, typing as t
from openllm_core._typing_compat import ParamSpec, Concatenate, TypeGuard
from openllm_core.exceptions import OpenLLMException
from openllm_core.utils import (
  apply,
  first_not_none,
  generate_hash_from_file,
  resolve_filepath,
  validate_is_path,
  normalise_model_name,
)

if t.TYPE_CHECKING:
  from bentoml import Model
  from .._llm import LLM

P = ParamSpec('P')
M = t.TypeVar('M')
T = t.TypeVar('T')

logger = logging.getLogger(__name__)


@apply(lambda val: tuple(str.lower(i) if i else i for i in val))
def _make_tag_components(model_id: str, model_version: t.Optional[str]) -> t.Tuple[str, t.Optional[str]]:
  model_id, *maybe_revision = model_id.rsplit(':')
  if len(maybe_revision) > 0:
    if model_version is not None:
      logger.warning(
        "revision is specified (%s). 'model_version=%s' will be ignored.", maybe_revision[0], model_version
      )
    model_version = maybe_revision[0]
  if validate_is_path(model_id):
    model_id = resolve_filepath(model_id)
    model_version = first_not_none(model_version, default=generate_hash_from_file(model_id))
  return normalise_model_name(model_id), model_version


def load_tokenizer(llm, **tokenizer_attrs):
  import cloudpickle, fs, transformers
  from bentoml._internal.models.model import CUSTOM_OBJECTS_FILENAME
  from .transformers._helpers import process_config

  tokenizer_attrs = {**llm.llm_parameters[-1], **tokenizer_attrs}

  config, *_ = process_config(llm.bentomodel.path, llm.trust_remote_code)

  bentomodel_fs = fs.open_fs(llm.bentomodel.path)
  if bentomodel_fs.isfile(CUSTOM_OBJECTS_FILENAME):
    with bentomodel_fs.open(CUSTOM_OBJECTS_FILENAME, 'rb') as cofile:
      try:
        tokenizer = cloudpickle.load(cofile)['tokenizer']
      except KeyError:
        raise OpenLLMException(
          "Bento model does not have tokenizer. Make sure to save the tokenizer within the model via 'custom_objects'. "
          'For example: "bentoml.transformers.save_model(..., custom_objects={\'tokenizer\': tokenizer})"'
        ) from None
  else:
    tokenizer = transformers.AutoTokenizer.from_pretrained(
      bentomodel_fs.getsyspath('/'), trust_remote_code=llm.trust_remote_code, **tokenizer_attrs
    )

  if tokenizer.pad_token_id is None:
    if config.pad_token_id is not None:
      tokenizer.pad_token_id = config.pad_token_id
    elif config.eos_token_id is not None:
      tokenizer.pad_token_id = config.eos_token_id
    elif tokenizer.eos_token_id is not None:
      tokenizer.pad_token_id = tokenizer.eos_token_id
  return tokenizer


def _make_dispatch_function(fn: str) -> t.Callable[Concatenate[LLM[M, T], P], TypeGuard[t.Union[M, T, Model]]]:
  def caller(llm: t.Optional[LLM[M, T]] = None, *args: P.args, **kwargs: P.kwargs) -> TypeGuard[t.Union[M, T, Model]]:
    """Generic function dispatch to correct serialisation submodules based on LLM runtime.

    > [!NOTE] See 'openllm.serialisation.transformers' if 'llm.__llm_backend__ in ("pt", "vllm")'

    > [!NOTE] See 'openllm.serialisation.ggml' if 'llm.__llm_backend__="ggml"'
    """
    backend = kwargs.get('_backend', None)
    if backend is None:
      if llm is None:
        raise OpenLLMException('Cannot dispatch without LLM instance.')
      backend = llm.__llm_backend__
    serde_mapping = {'pt': 'transformers', 'vllm': 'vllm', 'ggml': 'ggml'}
    try:
      serde = serde_mapping[backend]
    except KeyError:
      raise OpenLLMException(f'Not supported backend {backend}')
    call = getattr(importlib.import_module(f'.{serde}', 'openllm.serialisation'), fn)
    params = inspect.signature(call).parameters
    return call(llm, *args, **kwargs) if next(iter(params.keys())) == 'llm' else call(*args, **kwargs)

  return caller


_extras = ['import_model', 'load_model']
_import_structure = {'ggml', 'transformers', 'vllm', 'constants'}
__all__ = ['_make_tag_components', 'load_tokenizer', *_extras, *_import_structure]


def __dir__() -> t.Sequence[str]:
  return sorted(__all__)


def __getattr__(name: str) -> t.Any:
  if name in _import_structure:
    return importlib.import_module(f'.{name}', __name__)
  elif name in _extras:
    return _make_dispatch_function(name)
  else:
    raise AttributeError(f'{__name__} has no attribute {name}')
