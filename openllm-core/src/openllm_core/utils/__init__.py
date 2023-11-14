from __future__ import annotations
import contextlib
import functools
import hashlib
import logging
import logging.config
import os
import random
import socket
import sys
import types
import typing as t
import uuid
from pathlib import Path

from . import pkg
from .import_utils import ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES
from .lazy import LazyLoader as LazyLoader, LazyModule as LazyModule, VersionInfo as VersionInfo
from .._typing_compat import overload

if t.TYPE_CHECKING:
  from bentoml._internal.models.model import ModelContext
  from bentoml._internal.types import PathType
  from openllm_core._typing_compat import AnyCallable

DEBUG_ENV_VAR = 'BENTOML_DEBUG'
QUIET_ENV_VAR = 'BENTOML_QUIET'
# https://github.com/grpc/grpc/blob/master/doc/environment_variables.md
_GRPC_DEBUG_ENV_VAR = 'GRPC_VERBOSITY'

logger = logging.getLogger(__name__)

try:
  from typing import GenericAlias as _TypingGenericAlias  # type: ignore
except ImportError:
  # python < 3.9 does not have GenericAlias (list[int], tuple[str, ...] and so on)
  _TypingGenericAlias = ()  # type: ignore
if sys.version_info < (3, 10):
  _WithArgsTypes = (_TypingGenericAlias,)
else:
  #  _GenericAlias is the actual GenericAlias implementation
  _WithArgsTypes: t.Any = (t._GenericAlias, types.GenericAlias, types.UnionType)  # type: ignore

DEV_DEBUG_VAR = 'DEBUG'


def resolve_user_filepath(filepath: str, ctx: str | None) -> str:
  # Return if filepath exist after expanduser

  _path = os.path.expanduser(os.path.expandvars(filepath))
  if os.path.exists(_path):
    return os.path.realpath(_path)

  # Try finding file in ctx if provided
  if ctx:
    _path = os.path.expanduser(os.path.join(ctx, filepath))
    if os.path.exists(_path):
      return os.path.realpath(_path)

  raise FileNotFoundError(f'file {filepath} not found')


@contextlib.contextmanager
def reserve_free_port(
  host: str = 'localhost',
  port: int | None = None,
  prefix: str | None = None,
  max_retry: int = 50,
  enable_so_reuseport: bool = False,
) -> t.Iterator[int]:
  """
  detect free port and reserve until exit the context
  """
  import psutil

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  if enable_so_reuseport:
    if psutil.WINDOWS:
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    elif psutil.MACOS or psutil.FREEBSD:
      sock.setsockopt(socket.SOL_SOCKET, 0x10000, 1)  # SO_REUSEPORT_LB
    else:
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
      if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
        raise RuntimeError('Failed to set SO_REUSEPORT.') from None
  if prefix is not None:
    prefix_num = int(prefix) * 10 ** (5 - len(prefix))
    suffix_range = min(65535 - prefix_num, 10 ** (5 - len(prefix)))
    for _ in range(max_retry):
      suffix = random.randint(0, suffix_range)
      port = int(f'{prefix_num + suffix}')
      try:
        sock.bind((host, port))
        break
      except OSError:
        continue
    else:
      raise RuntimeError(f'Cannot find free port with prefix {prefix} after {max_retry} retries.') from None
  elif port:
    sock.bind((host, port))
  else:
    sock.bind((host, 0))
  try:
    yield sock.getsockname()[1]
  finally:
    sock.close()


def calc_dir_size(path: PathType) -> int:
  return sum(f.stat().st_size for f in Path(path).glob('**/*') if f.is_file())


def set_debug_mode(enabled: bool, level: int = 1) -> None:
  # monkeypatch bentoml._internal.configuration.set_debug_mode to remove unused logs
  if enabled:
    os.environ[DEV_DEBUG_VAR] = str(level)
  os.environ[DEBUG_ENV_VAR] = str(enabled)
  os.environ[_GRPC_DEBUG_ENV_VAR] = 'DEBUG' if enabled else 'ERROR'
  set_disable_warnings(enabled)


def lenient_issubclass(cls: t.Any, class_or_tuple: type[t.Any] | tuple[type[t.Any], ...] | None) -> bool:
  try:
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)  # type: ignore[arg-type]
  except TypeError:
    if isinstance(cls, _WithArgsTypes):
      return False
    raise


@functools.lru_cache(maxsize=128)
def generate_hash_from_file(f: str, algorithm: t.Literal['md5', 'sha1'] = 'sha1') -> str:
  """Generate a hash from given file's modification time.

  Args:
  f: The file to generate the hash from.
  algorithm: The hashing algorithm to use. Defaults to 'sha1' (similar to how Git generate its commit hash.)

  Returns:
  The generated hash.
  """
  return getattr(hashlib, algorithm)(str(os.path.getmtime(resolve_filepath(f))).encode()).hexdigest()


def check_bool_env(env: str, default: bool = True) -> bool:
  v = os.environ.get(env, str(default)).upper()
  if v.isdigit():
    return bool(int(v))  # special check for digits
  return v in ENV_VARS_TRUE_VALUES


# equivocal setattr to save one lookup per assignment
_object_setattr = object.__setattr__


def field_env_key(key: str, suffix: str | None = None) -> str:
  return '_'.join(filter(None, map(str.upper, ['OPENLLM', suffix.strip('_') if suffix else '', key])))


# Special debug flag controled via DEBUG
DEBUG: bool = sys.flags.dev_mode or (not sys.flags.ignore_environment and check_bool_env(DEV_DEBUG_VAR, default=False))
# Whether to show the codenge for debug purposes
SHOW_CODEGEN: bool = DEBUG and (
  os.environ.get(DEV_DEBUG_VAR, str(0)).isdigit() and int(os.environ.get(DEV_DEBUG_VAR, str(0))) > 3
)
# MYPY is like t.TYPE_CHECKING, but reserved for Mypy plugins
MYPY = False


def get_debug_mode() -> bool:
  if not DEBUG and DEBUG_ENV_VAR in os.environ:
    return check_bool_env(DEBUG_ENV_VAR, False)
  return DEBUG


def get_quiet_mode() -> bool:
  if QUIET_ENV_VAR in os.environ:
    return check_bool_env(QUIET_ENV_VAR, False)
  if DEBUG:
    return False
  return False


def set_quiet_mode(enabled: bool) -> None:
  # do not log setting quiet mode
  os.environ[QUIET_ENV_VAR] = str(enabled)
  os.environ[_GRPC_DEBUG_ENV_VAR] = 'NONE'
  set_disable_warnings(enabled)


WARNING_ENV_VAR = 'OPENLLM_DISABLE_WARNING'


def get_disable_warnings() -> bool:
  if get_debug_mode():
    return False
  return check_bool_env(WARNING_ENV_VAR, False)


def set_disable_warnings(disable: bool = True) -> None:
  if get_disable_warnings():
    os.environ[WARNING_ENV_VAR] = str(disable)


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


def gen_random_uuid(prefix: str | None = None) -> str:
  return '-'.join([prefix or 'openllm', str(uuid.uuid4().hex)])


_LOGGING_CONFIG: dict[str, t.Any] = {
  'version': 1,
  'disable_existing_loggers': True,
  'filters': {
    'excfilter': {'()': 'openllm_core.utils.ExceptionFilter'},
    'infofilter': {'()': 'openllm_core.utils.InfoFilter'},
  },
  'handlers': {
    'bentomlhandler': {
      'class': 'logging.StreamHandler',
      'filters': ['excfilter', 'infofilter'],
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


def configure_logging() -> None:
  """Configure logging for OpenLLM.

  Behaves similar to how BentoML loggers are being configured.
  """
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

  logging.config.dictConfig(_LOGGING_CONFIG)


@functools.lru_cache(maxsize=1)
def in_notebook() -> bool:
  try:
    from IPython.core.getipython import get_ipython

    if t.TYPE_CHECKING:
      from IPython.core.interactiveshell import InteractiveShell
    return 'IPKernelApp' in t.cast(
      'dict[str, t.Any]', t.cast(t.Callable[[], 'InteractiveShell'], get_ipython)().config
    )
  except (ImportError, AttributeError):
    return False


class suppress(contextlib.suppress, contextlib.ContextDecorator):
  """A version of contextlib.suppress with decorator support.

  >>> @suppress(KeyError)
  ... def key_error():
  ...     {}['']
  >>> key_error()
  """


def compose(*funcs: AnyCallable) -> AnyCallable:
  """Compose any number of unary functions into a single unary function.

  >>> import textwrap
  >>> expected = str.strip(textwrap.dedent(compose.__doc__))
  >>> strip_and_dedent = compose(str.strip, textwrap.dedent)
  >>> strip_and_dedent(compose.__doc__) == expected
  True

  Compose also allows the innermost function to take arbitrary arguments.

  >>> round_three = lambda x: round(x, ndigits=3)
  >>> f = compose(round_three, int.__truediv__)
  >>> [f(3*x, x+1) for x in range(1,10)]
  [1.5, 2.0, 2.25, 2.4, 2.5, 2.571, 2.625, 2.667, 2.7]
  """

  def compose_two(f1: AnyCallable, f2: AnyCallable) -> AnyCallable:
    return lambda *args, **kwargs: f1(f2(*args, **kwargs))

  return functools.reduce(compose_two, funcs)


def apply(transform: AnyCallable) -> t.Callable[[AnyCallable], AnyCallable]:
  """Decorate a function with a transform function that is invoked on results returned from the decorated function.

  ```python
  @apply(reversed)
  def get_numbers(start):
  "doc for get_numbers"
  return range(start, start+3)
  list(get_numbers(4))
  # [6, 5, 4]
  ```
  ```python
  get_numbers.__doc__
  # 'doc for get_numbers'
  ```
  """
  return lambda func: functools.wraps(func)(compose(transform, func))


T = t.TypeVar('T')
K = t.TypeVar('K')


# yapf: disable
@overload
def first_not_none(*args: T | None, default: T) -> T: ...
@overload
def first_not_none(*args: T | None) -> T | None: ...
def first_not_none(*args: T | None, default: None | T = None) -> T | None: return next((arg for arg in args if arg is not None), default)
# yapf: enable


def resolve_filepath(path: str, ctx: str | None = None) -> str:
  """Resolve a file path to an absolute path, expand user and environment variables."""
  try:
    return resolve_user_filepath(path, ctx)
  except FileNotFoundError:
    return path


def validate_is_path(maybe_path: str) -> bool:
  return os.path.exists(os.path.dirname(resolve_filepath(maybe_path)))


def generate_context(framework_name: str) -> ModelContext:
  import openllm_core
  from bentoml._internal.models.model import ModelContext

  framework_versions = {'transformers': pkg.get_pkg_version('transformers')}
  if openllm_core.utils.is_torch_available():
    framework_versions['torch'] = pkg.get_pkg_version('torch')
  return ModelContext(framework_name=framework_name, framework_versions=framework_versions)


_TOKENIZER_PREFIX = '_tokenizer_'


def flatten_attrs(**attrs: t.Any) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
  """Normalize the given attrs to a model and tokenizer kwargs accordingly."""
  tokenizer_attrs = {k[len(_TOKENIZER_PREFIX) :]: v for k, v in attrs.items() if k.startswith(_TOKENIZER_PREFIX)}
  for k in tuple(attrs.keys()):
    if k.startswith(_TOKENIZER_PREFIX):
      del attrs[k]
  return attrs, tokenizer_attrs


# NOTE: The set marks contains a set of modules name
# that are available above and are whitelisted
# to be included in the extra_objects map.
_whitelist_modules = {'pkg'}

# XXX: define all classes, functions import above this line
# since _extras will be the locals() import from this file.
_extras: dict[str, t.Any] = {
  k: v
  for k, v in locals().items()
  if k in _whitelist_modules or (not isinstance(v, types.ModuleType) and not k.startswith('_'))
}
_extras['__openllm_migration__'] = {'bentoml_cattr': 'converter'}
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
      'is_transformers_available',
      'is_optimum_supports_gptq',
      'is_autoawq_available',
      'is_bentoml_available',
    ],
  },
  extra_objects=_extras,
)
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__

if t.TYPE_CHECKING:
  # NOTE: The following exports useful utils from bentoml
  from . import analytics as analytics, codegen as codegen, dantic as dantic, serde as serde
  from .import_utils import (
    OPTIONAL_DEPENDENCIES as OPTIONAL_DEPENDENCIES,
    is_autoawq_available as is_autoawq_available,
    is_autogptq_available as is_autogptq_available,
    is_bentoml_available as is_bentoml_available,
    is_bitsandbytes_available as is_bitsandbytes_available,
    is_grpc_available as is_grpc_available,
    is_jupyter_available as is_jupyter_available,
    is_jupytext_available as is_jupytext_available,
    is_notebook_available as is_notebook_available,
    is_optimum_supports_gptq as is_optimum_supports_gptq,
    is_peft_available as is_peft_available,
    is_torch_available as is_torch_available,
    is_transformers_available as is_transformers_available,
    is_vllm_available as is_vllm_available,
  )
  from .representation import ReprMixin as ReprMixin
  from .serde import converter as converter
