from __future__ import annotations
import functools, hashlib, logging, logging.config, pydantic, inflection, os, sys, types, uuid, typing as t
from pathlib import Path as _Path
from . import import_utils as iutils, pkg
from .lazy import LazyLoader as LazyLoader, LazyModule as LazyModule, VersionInfo as VersionInfo
from ._constants import (
  DEBUG_ENV_VAR as DEBUG_ENV_VAR,
  QUIET_ENV_VAR as QUIET_ENV_VAR,
  GRPC_DEBUG_ENV_VAR as GRPC_DEBUG_ENV_VAR,
  WARNING_ENV_VAR as WARNING_ENV_VAR,
  DEV_DEBUG_VAR as DEV_DEBUG_VAR,
  ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES,
  check_bool_env as check_bool_env,
  DEBUG as DEBUG,
  SHOW_CODEGEN as SHOW_CODEGEN,
  MYPY as MYPY,
  OPENLLM_DEV_BUILD as OPENLLM_DEV_BUILD,
)

if t.TYPE_CHECKING:
  from _bentoml_sdk import IODescriptor

# equivocal setattr to save one lookup per assignment
_object_setattr = object.__setattr__
logger = logging.getLogger(__name__)


def normalise_model_name(name):
  return (
    os.path.basename(resolve_filepath(name))
    if validate_is_path(name)
    else inflection.dasherize(name.replace('/', '--'))
  )


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


def correct_closure(cls, ref):
  # The following is a fix for
  # <https://github.com/python-attrs/attrs/issues/102>.
  # If a method mentions `__class__` or uses the no-arg super(), the
  # compiler will bake a reference to the class in the method itself
  # as `method.__closure__`.  Since we replace the class with a
  # clone, we rewrite these references so it keeps working.
  for item in cls.__dict__.values():
    if isinstance(item, (classmethod, staticmethod)):
      # Class- and staticmethods hide their functions inside.
      # These might need to be rewritten as well.
      closure_cells = getattr(item.__func__, '__closure__', None)
    elif isinstance(item, property):
      # Workaround for property `super()` shortcut (PY3-only).
      # There is no universal way for other descriptors.
      closure_cells = getattr(item.fget, '__closure__', None)
    else:
      closure_cells = getattr(item, '__closure__', None)

    if not closure_cells:
      continue  # Catch None or the empty list.
    for cell in closure_cells:
      try:
        match = cell.cell_contents is ref
      except ValueError:  # noqa: PERF203
        pass  # ValueError: Cell is empty
      else:
        if match:
          cell.cell_contents = cls
  return cls


@functools.lru_cache(maxsize=1)
def _WithArgsTypes() -> tuple[type[t.Any], ...]:
  try:
    from typing import GenericAlias as _TypingGenericAlias
  except ImportError:
    _TypingGenericAlias = ()  # python < 3.9 does not have GenericAlias (list[int], tuple[str, ...] and so on)
  #  _GenericAlias is the actual GenericAlias implementation
  return (
    (_TypingGenericAlias,) if sys.version_info < (3, 10) else (t._GenericAlias, types.GenericAlias, types.UnionType)
  )


def lenient_issubclass(cls, class_or_tuple):
  try:
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)
  except TypeError:
    if isinstance(cls, _WithArgsTypes()):
      return False
    raise


def io_descriptor(model) -> type[IODescriptor] | None:
  if model is None:
    return model
  try:
    from _bentoml_sdk.io_models import IODescriptor
  except ImportError as err:
    raise RuntimeError('Requires "bentoml>1.2" to use `openllm_core.utils.io_descriptor`') from err

  return pydantic.create_model(f'{model.__class__.__name__}IODescriptor', __base__=(IODescriptor, model))


def api(
  func=None,
  *,
  name=None,
  input=None,
  output=None,
  route=None,
  batchable=False,
  batch_dim=0,
  max_batch_size=100,
  max_latency_ms=60000,
):
  try:
    import bentoml
  except ImportError:
    raise RuntimeError('Requires "bentoml" to use `openllm_core.utils.api`') from None

  def caller(func):
    wrapped = bentoml.api(
      func,
      route=route,
      name=name,
      input_spec=io_descriptor(input),
      output_spec=io_descriptor(output),
      batchable=batchable,
      batch_dim=batch_dim,
      max_batch_size=max_batch_size,
      max_latency_ms=max_latency_ms,
    )
    object.__setattr__(func, '__openllm_api_func__', True)
    return wrapped

  return caller(func) if func is not None else caller


def resolve_user_filepath(filepath, ctx):
  _path = os.path.expanduser(os.path.expandvars(filepath))
  if os.path.exists(_path):
    return os.path.realpath(_path)
  # Try finding file in ctx if provided
  if ctx:
    _path = os.path.expanduser(os.path.join(ctx, filepath))
    if os.path.exists(_path):
      return os.path.realpath(_path)
  raise FileNotFoundError(f'file {filepath} not found')


# this is the supress version of resolve_user_filepath
def resolve_filepath(path, ctx=None):
  try:
    return resolve_user_filepath(path, ctx)
  except FileNotFoundError:
    return path


def calc_dir_size(path):
  return sum(f.stat().st_size for f in _Path(path).glob('**/*') if f.is_file())


@functools.lru_cache(maxsize=128)
def generate_hash_from_file(f, algorithm='sha1'):
  return str(getattr(hashlib, algorithm)(str(os.path.getmtime(resolve_filepath(f))).encode()).hexdigest())


def getenv(env, default=None, var=None, return_type=t.Any):
  env_key = {env.upper(), f'OPENLLM_{env.upper()}'}
  if var is not None:
    env_key = set(var) | env_key

  def callback(k: str) -> t.Any:
    _var = os.getenv(k)
    if _var and k.startswith('OPENLLM_'):
      logger.warning("Using '%s' environment is deprecated, use '%s' instead.", k.upper(), k[8:].upper())
    return _var

  return t.cast(return_type, first_not_none(*(callback(k) for k in env_key), default=default))


def field_env_key(key, suffix=None):
  return '_'.join(filter(None, map(str.upper, ['OPENLLM', suffix.strip('_') if suffix else '', key])))


def get_debug_mode():
  return check_bool_env(DEBUG_ENV_VAR, False) if (not DEBUG and DEBUG_ENV_VAR in os.environ) else DEBUG


def get_quiet_mode():
  if QUIET_ENV_VAR in os.environ:
    return check_bool_env(QUIET_ENV_VAR, False)
  if DEBUG:
    return False
  return False


def get_disable_warnings():
  return check_bool_env(WARNING_ENV_VAR, False)


def set_disable_warnings(disable=True):
  if disable:
    os.environ[WARNING_ENV_VAR] = str(disable)


def set_debug_mode(enabled, level=1):
  if enabled:
    os.environ[DEV_DEBUG_VAR] = str(level)
  os.environ.update({
    DEBUG_ENV_VAR: str(enabled),
    QUIET_ENV_VAR: str(not enabled),  #
    GRPC_DEBUG_ENV_VAR: 'DEBUG' if enabled else 'ERROR',
    'CT2_VERBOSE': '3',  #
  })
  set_disable_warnings(not enabled)


def set_quiet_mode(enabled):
  os.environ.update({
    QUIET_ENV_VAR: str(enabled),
    DEBUG_ENV_VAR: str(not enabled),  #
    GRPC_DEBUG_ENV_VAR: 'NONE',
    'CT2_VERBOSE': '-1',  #
  })
  set_disable_warnings(enabled)


def gen_random_uuid(prefix: str | None = None) -> str:
  return '-'.join([prefix or 'openllm', str(uuid.uuid4().hex)])


def dict_filter_none(d: dict[str, t.Any]) -> dict[str, t.Any]:
  return {k: v for k, v in d.items() if v is not None}


# NOTE:  `compose` any number of unary functions into a single unary function
# compose(f, g, h)(x) == f(g(h(x))); compose(f, g, h)(x, y, z) == f(g(h(x, y, z)))
def compose(*funcs):
  return functools.reduce(lambda f1, f2: lambda *args, **kwargs: f1(f2(*args, **kwargs)), funcs)


# NOTE:  `apply` a transform function that is invoked on results returned from the decorated function
# apply(reversed)(func)(*args, **kwargs) == reversed(func(*args, **kwargs))
def apply(transform):
  return lambda func: functools.wraps(func)(compose(transform, func))


def validate_is_path(maybe_path):
  return os.path.exists(os.path.dirname(resolve_filepath(maybe_path)))


def first_not_none(*args, default=None):
  return next((arg for arg in args if arg is not None), default)


def generate_context(framework_name):
  from bentoml._internal.models.model import ModelContext

  framework_versions = {
    'transformers': pkg.get_pkg_version('transformers'),
    'safetensors': pkg.get_pkg_version('safetensors'),
    'optimum': pkg.get_pkg_version('optimum'),
    'accelerate': pkg.get_pkg_version('accelerate'),
  }
  if iutils.is_torch_available():
    framework_versions['torch'] = pkg.get_pkg_version('torch')
  if iutils.is_vllm_available():
    framework_versions['vllm'] = pkg.get_pkg_version('vllm')
  if iutils.is_autoawq_available():
    framework_versions['autoawq'] = pkg.get_pkg_version('autoawq')
  if iutils.is_autogptq_available():
    framework_versions['autogptq'] = pkg.get_pkg_version('auto_gptq')
  if iutils.is_bentoml_available():
    framework_versions['bentoml'] = pkg.get_pkg_version('bentoml')
  if iutils.is_triton_available():
    framework_versions['triton'] = pkg.get_pkg_version('triton')
  if iutils.is_flash_attn_2_available():
    framework_versions['flash_attn'] = pkg.get_pkg_version('flash_attn')
  return ModelContext(framework_name=framework_name, framework_versions=framework_versions)


@functools.lru_cache(maxsize=1)
def in_notebook():
  try:
    from IPython.core.getipython import get_ipython

    return 'IPKernelApp' in get_ipython().config
  except Exception:
    return False


def flatten_attrs(**attrs):
  _TOKENIZER_PREFIX = '_tokenizer_'
  tokenizer_attrs = {k[len(_TOKENIZER_PREFIX) :]: v for k, v in attrs.items() if k.startswith(_TOKENIZER_PREFIX)}
  for k in tuple(attrs.keys()):
    if k.startswith(_TOKENIZER_PREFIX):
      del attrs[k]
  return attrs, tokenizer_attrs


class ExceptionFilter(logging.Filter):
  def __init__(self, exclude_exceptions: list[type[Exception]] | None = None, **kwargs: t.Any):
    if exclude_exceptions is None:
      exclude_exceptions = []
    try:
      from circus.exc import ConflictError

      if ConflictError not in exclude_exceptions:
        exclude_exceptions.append(ConflictError)
    except ImportError:
      pass
    super(ExceptionFilter, self).__init__(**kwargs)
    self.EXCLUDE_EXCEPTIONS = exclude_exceptions

  def filter(self, record: logging.LogRecord) -> bool:
    if record.exc_info:
      etype, _, _ = record.exc_info
      if etype is not None:
        for exc in self.EXCLUDE_EXCEPTIONS:
          if issubclass(etype, exc):
            return False
    return True


class InfoFilter(logging.Filter):
  def filter(self, record: logging.LogRecord) -> bool:
    return logging.INFO <= record.levelno < logging.WARNING


class WarningFilter(logging.Filter):  # FIXME: Why does this not work?
  def filter(self, record: logging.LogRecord) -> bool:
    if get_disable_warnings():
      return record.levelno >= logging.ERROR
    return True


_LOGGING_CONFIG = {
  'version': 1,
  'disable_existing_loggers': True,
  'filters': {
    'excfilter': {'()': 'openllm_core.utils.ExceptionFilter'},
    'infofilter': {'()': 'openllm_core.utils.InfoFilter'},
    'warningfilter': {'()': 'openllm_core.utils.WarningFilter'},
  },
  'handlers': {
    'bentomlhandler': {
      'class': 'logging.StreamHandler',
      'filters': ['excfilter', 'warningfilter', 'infofilter'],
      'stream': 'ext://sys.stdout',
    },
    'defaulthandler': {'class': 'logging.StreamHandler', 'level': logging.WARNING},
  },
  'loggers': {
    'bentoml': {'handlers': ['bentomlhandler', 'defaulthandler'], 'level': logging.INFO, 'propagate': False},
    'openllm': {'handlers': ['bentomlhandler', 'defaulthandler'], 'level': logging.INFO, 'propagate': False},
  },
  'root': {'level': logging.WARNING},
}


class Counter:
  def __init__(self, start: int = 0):
    self.counter = start

  def __next__(self) -> int:
    i = self.counter
    self.counter += 1
    return i

  def reset(self) -> None:
    self.counter = 0


def configure_logging():
  if get_quiet_mode():
    _LOGGING_CONFIG['loggers']['openllm']['level'] = logging.ERROR
    _LOGGING_CONFIG['loggers']['bentoml']['level'] = logging.ERROR
    _LOGGING_CONFIG['root']['level'] = logging.ERROR
  elif get_debug_mode() or DEBUG:
    _LOGGING_CONFIG['handlers']['defaulthandler']['level'] = logging.DEBUG
    _LOGGING_CONFIG['loggers']['openllm']['level'] = logging.DEBUG
    _LOGGING_CONFIG['loggers']['bentoml']['level'] = logging.DEBUG
    _LOGGING_CONFIG['root']['level'] = logging.DEBUG
  else:
    _LOGGING_CONFIG['loggers']['openllm']['level'] = logging.INFO
    _LOGGING_CONFIG['loggers']['bentoml']['level'] = logging.INFO
    _LOGGING_CONFIG['root']['level'] = logging.INFO

  if get_disable_warnings():  # HACK: This is a hack to disable warnings
    _LOGGING_CONFIG['loggers']['openllm']['level'] = logging.ERROR

  logging.config.dictConfig(_LOGGING_CONFIG)


# XXX: define all classes, functions import above this line
# since _extras will be the locals() import from this file.
_extras = {
  **{
    k: v
    for k, v in locals().items()
    if k in {'pkg'} or (not isinstance(v, types.ModuleType) and k not in {'annotations'} and not k.startswith('_'))
  },
  '__openllm_migration__': {'bentoml_cattr': 'converter'},
}
__lazy = LazyModule(
  __name__,
  globals()['__file__'],
  {
    'analytics': [],
    'codegen': [],
    'dantic': [],
    'lazy': [],
    'pkg': [],
    'representation': ['ReprMixin'],
    'serde': ['converter'],
    'import_utils': [
      'is_vllm_available',
      'is_torch_available',
      'is_bitsandbytes_available',
      'is_peft_available',
      'is_jupyter_available',
      'is_jupytext_available',
      'is_notebook_available',
      'is_autogptq_available',
      'is_grpc_available',
      'is_transformers_available',
      'is_autoawq_available',
      'is_flash_attn_2_available',
      'is_bentoml_available',
      'is_triton_available',
    ],
  },
  extra_objects=_extras,
)
__all__, __dir__, __getattr__ = __lazy.__all__, __lazy.__dir__, __lazy.__getattr__
