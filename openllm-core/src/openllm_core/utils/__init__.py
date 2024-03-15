from __future__ import annotations
import functools, hashlib, logging, logging.config, os, sys, types, uuid, typing as t
from pathlib import Path as _Path
from . import import_utils as iutils, pkg
from .import_utils import ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES
from .lazy import LazyLoader as LazyLoader, LazyModule as LazyModule, VersionInfo as VersionInfo

# See https://github.com/bentoml/BentoML/blob/a59750c5044bab60b6b3765e6c17041fd8984712/src/bentoml_cli/env.py#L17
DEBUG_ENV_VAR = 'BENTOML_DEBUG'
QUIET_ENV_VAR = 'BENTOML_QUIET'
# https://github.com/grpc/grpc/blob/master/doc/environment_variables.md
_GRPC_DEBUG_ENV_VAR = 'GRPC_VERBOSITY'
WARNING_ENV_VAR = 'OPENLLM_DISABLE_WARNING'
DEV_DEBUG_VAR = 'DEBUG'
# equivocal setattr to save one lookup per assignment
_object_setattr = object.__setattr__
logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _WithArgsTypes() -> tuple[type[t.Any], ...]:
  try:
    from typing import GenericAlias as _TypingGenericAlias
  except ImportError:
    _TypingGenericAlias = ()  # python < 3.9 does not have GenericAlias (list[int], tuple[str, ...] and so on)
  #  _GenericAlias is the actual GenericAlias implementation
  return (_TypingGenericAlias,) if sys.version_info < (3, 10) else (t._GenericAlias, types.GenericAlias, types.UnionType)


def lenient_issubclass(cls, class_or_tuple):
  try:
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)
  except TypeError:
    if isinstance(cls, _WithArgsTypes()):
      return False
    raise


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


def check_bool_env(env, default=True):
  v = os.getenv(env, default=str(default)).upper()
  if v.isdigit():
    return bool(int(v))  # special check for digits
  return v in ENV_VARS_TRUE_VALUES


def calc_dir_size(path):
  return sum(f.stat().st_size for f in _Path(path).glob('**/*') if f.is_file())


@functools.lru_cache(maxsize=128)
def generate_hash_from_file(f, algorithm='sha1'):
  return str(getattr(hashlib, algorithm)(str(os.path.getmtime(resolve_filepath(f))).encode()).hexdigest())


def getenv(env, default=None, var=None):
  env_key = {env.upper(), f'OPENLLM_{env.upper()}'}
  if var is not None:
    env_key = set(var) | env_key

  def callback(k: str) -> t.Any:
    _var = os.getenv(k)
    if _var and k.startswith('OPENLLM_'):
      logger.warning("Using '%s' environment is deprecated, use '%s' instead.", k.upper(), k[8:].upper())
    return _var

  return first_not_none(*(callback(k) for k in env_key), default=default)


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
    _GRPC_DEBUG_ENV_VAR: 'DEBUG' if enabled else 'ERROR',
    'CT2_VERBOSE': '3',  #
  })
  set_disable_warnings(not enabled)


def set_quiet_mode(enabled):
  os.environ.update({
    QUIET_ENV_VAR: str(enabled),
    DEBUG_ENV_VAR: str(not enabled),  #
    _GRPC_DEBUG_ENV_VAR: 'NONE',
    'CT2_VERBOSE': '-1',  #
  })
  set_disable_warnings(enabled)


def gen_random_uuid(prefix: str | None = None) -> str:
  return '-'.join([prefix or 'openllm', str(uuid.uuid4().hex)])


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
  if iutils.is_ctranslate_available():
    framework_versions['ctranslate2'] = pkg.get_pkg_version('ctranslate2')
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


# Special debug flag controled via DEBUG
DEBUG = sys.flags.dev_mode or (not sys.flags.ignore_environment and check_bool_env(DEV_DEBUG_VAR, default=False))
# Whether to show the codenge for debug purposes
SHOW_CODEGEN = DEBUG and os.environ.get(DEV_DEBUG_VAR, str(0)).isdigit() and int(os.environ.get(DEV_DEBUG_VAR, str(0))) > 3
# MYPY is like t.TYPE_CHECKING, but reserved for Mypy plugins
MYPY = False


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
    'bentomlhandler': {'class': 'logging.StreamHandler', 'filters': ['excfilter', 'warningfilter', 'infofilter'], 'stream': 'ext://sys.stdout'},
    'defaulthandler': {'class': 'logging.StreamHandler', 'level': logging.WARNING},
  },
  'loggers': {
    'bentoml': {'handlers': ['bentomlhandler', 'defaulthandler'], 'level': logging.INFO, 'propagate': False},
    'openllm': {'handlers': ['bentomlhandler', 'defaulthandler'], 'level': logging.INFO, 'propagate': False},
  },
  'root': {'level': logging.WARNING},
}


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
    k: v for k, v in locals().items() if k in {'pkg'} or (not isinstance(v, types.ModuleType) and k not in {'annotations'} and not k.startswith('_'))
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
      'OPTIONAL_DEPENDENCIES',
      'is_vllm_available',
      'is_torch_available',
      'is_bitsandbytes_available',
      'is_peft_available',
      'is_jupyter_available',
      'is_jupytext_available',
      'is_notebook_available',
      'is_autogptq_available',
      'is_grpc_available',
      'is_ctranslate_available',
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
