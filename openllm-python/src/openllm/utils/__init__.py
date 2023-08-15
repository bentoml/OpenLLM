"""Utilities function for OpenLLM.

User can import these function for convenience, but
we won't ensure backward compatibility for these functions. So use with caution.
"""
from __future__ import annotations
import contextlib, functools, hashlib, logging, logging.config, os, sys, types, typing as t, openllm
from pathlib import Path
from circus.exc import ConflictError
from bentoml._internal.configuration import (
  DEBUG_ENV_VAR as DEBUG_ENV_VAR,
  GRPC_DEBUG_ENV_VAR as _GRPC_DEBUG_ENV_VAR,
  QUIET_ENV_VAR as QUIET_ENV_VAR,
  get_debug_mode as _get_debug_mode,
  get_quiet_mode as _get_quiet_mode,
  set_quiet_mode as set_quiet_mode,
)
from bentoml._internal.models.model import ModelContext as _ModelContext
from bentoml._internal.types import LazyType as LazyType
from bentoml._internal.utils import (
  LazyLoader as LazyLoader,
  bentoml_cattr as bentoml_cattr,
  calc_dir_size as calc_dir_size,
  first_not_none as first_not_none,
  pkg as pkg,
  reserve_free_port as reserve_free_port,
  resolve_user_filepath as resolve_user_filepath,
)
from openllm.utils.lazy import (
  LazyModule as LazyModule,
  VersionInfo as VersionInfo,
)

if t.TYPE_CHECKING:
  from openllm._typing_compat import AnyCallable, LiteralRuntime

logger = logging.getLogger(__name__)
try: from typing import GenericAlias as _TypingGenericAlias  # type: ignore
except ImportError: _TypingGenericAlias = ()  # type: ignore # python < 3.9 does not have GenericAlias (list[int], tuple[str, ...] and so on)
if sys.version_info < (3, 10): _WithArgsTypes = (_TypingGenericAlias,)
else: _WithArgsTypes: t.Any = (t._GenericAlias, types.GenericAlias, types.UnionType)  # type: ignore #  _GenericAlias is the actual GenericAlias implementation

DEV_DEBUG_VAR = "OPENLLMDEVDEBUG"

def set_debug_mode(enabled: bool, level: int = 1) -> None:
  # monkeypatch bentoml._internal.configuration.set_debug_mode to remove unused logs
  if enabled: os.environ[DEV_DEBUG_VAR] = str(level)
  os.environ[DEBUG_ENV_VAR] = str(enabled)
  os.environ[_GRPC_DEBUG_ENV_VAR] = "DEBUG" if enabled else "ERROR"

def lenient_issubclass(cls: t.Any, class_or_tuple: type[t.Any] | tuple[type[t.Any], ...] | None) -> bool:
  try:
    return isinstance(cls, type) and issubclass(cls, class_or_tuple)  # type: ignore[arg-type]
  except TypeError:
    if isinstance(cls, _WithArgsTypes): return False
    raise

def available_devices() -> tuple[str, ...]:
  """Return available GPU under system. Currently only supports NVIDIA GPUs."""
  from openllm._strategies import NvidiaGpuResource
  return tuple(NvidiaGpuResource.from_system())

@functools.lru_cache(maxsize=128)
def generate_hash_from_file(f: str, algorithm: t.Literal["md5", "sha1"] = "sha1") -> str:
  """Generate a hash from given file's modification time.

  Args:
  f: The file to generate the hash from.
  algorithm: The hashing algorithm to use. Defaults to 'sha1' (similar to how Git generate its commit hash.)

  Returns:
  The generated hash.
  """
  return getattr(hashlib, algorithm)(str(os.path.getmtime(resolve_filepath(f))).encode()).hexdigest()

@functools.lru_cache(maxsize=1)
def device_count() -> int: return len(available_devices())

# equivocal setattr to save one lookup per assignment
_object_setattr = object.__setattr__

def non_intrusive_setattr(obj: t.Any, name: str, value: t.Any) -> None:
  """This makes sure that we don't overwrite any existing attributes on the object."""
  _setattr = functools.partial(setattr, obj) if isinstance(obj, type) else _object_setattr.__get__(obj)
  if not hasattr(obj, name): _setattr(name, value)

def field_env_key(model_name: str, key: str, suffix: str | None = None) -> str: return "_".join(filter(None, map(str.upper, ["OPENLLM", model_name, suffix.strip("_") if suffix else "", key])))

# Special debug flag controled via OPENLLMDEVDEBUG
DEBUG: bool = sys.flags.dev_mode or (not sys.flags.ignore_environment and bool(os.environ.get(DEV_DEBUG_VAR)))
# MYPY is like t.TYPE_CHECKING, but reserved for Mypy plugins
MYPY = False
SHOW_CODEGEN: bool = DEBUG and int(os.environ.get("OPENLLMDEVDEBUG", str(0))) > 3

def get_debug_mode() -> bool: return DEBUG or _get_debug_mode()
def get_quiet_mode() -> bool: return not DEBUG and _get_quiet_mode()

class ExceptionFilter(logging.Filter):
  def __init__(self, exclude_exceptions: list[type[Exception]] | None = None, **kwargs: t.Any):
    """A filter of all exception."""
    if exclude_exceptions is None: exclude_exceptions = [ConflictError]
    if ConflictError not in exclude_exceptions: exclude_exceptions.append(ConflictError)
    super(ExceptionFilter, self).__init__(**kwargs)
    self.EXCLUDE_EXCEPTIONS = exclude_exceptions

  def filter(self, record: logging.LogRecord) -> bool:
    if record.exc_info:
      etype, _, _ = record.exc_info
      if etype is not None:
        for exc in self.EXCLUDE_EXCEPTIONS:
          if issubclass(etype, exc): return False
    return True

class InfoFilter(logging.Filter):
  def filter(self, record: logging.LogRecord) -> bool: return logging.INFO <= record.levelno < logging.WARNING

_LOGGING_CONFIG: dict[str, t.Any] = {
    "version": 1, "disable_existing_loggers": True,
    "filters": {"excfilter": {"()": "openllm.utils.ExceptionFilter"}, "infofilter": {"()": "openllm.utils.InfoFilter"}},
    "handlers": {"bentomlhandler": {"class": "logging.StreamHandler", "filters": ["excfilter", "infofilter"], "stream": "ext://sys.stdout"}, "defaulthandler": {"class": "logging.StreamHandler", "level": logging.WARNING}},
    "loggers": {"bentoml": {"handlers": ["bentomlhandler", "defaulthandler"], "level": logging.INFO, "propagate": False}, "openllm": {"handlers": ["bentomlhandler", "defaulthandler"], "level": logging.INFO, "propagate": False,}},
    "root": {"level": logging.WARNING},
}

def configure_logging() -> None:
  """Configure logging for OpenLLM.

  Behaves similar to how BentoML loggers are being configured.
  """
  if get_quiet_mode():
    _LOGGING_CONFIG["loggers"]["openllm"]["level"] = logging.ERROR
    _LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.ERROR
    _LOGGING_CONFIG["root"]["level"] = logging.ERROR
  elif get_debug_mode() or DEBUG:
    _LOGGING_CONFIG["loggers"]["openllm"]["level"] = logging.DEBUG
    _LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.DEBUG
    _LOGGING_CONFIG["root"]["level"] = logging.DEBUG
  else:
    _LOGGING_CONFIG["loggers"]["openllm"]["level"] = logging.INFO
    _LOGGING_CONFIG["loggers"]["bentoml"]["level"] = logging.INFO
    _LOGGING_CONFIG["root"]["level"] = logging.INFO

  logging.config.dictConfig(_LOGGING_CONFIG)

@functools.lru_cache(maxsize=1)
def in_notebook() -> bool:
  try:
    from IPython.core.getipython import get_ipython
    if t.TYPE_CHECKING:
      from IPython.core.interactiveshell import InteractiveShell
    return "IPKernelApp" in t.cast("dict[str, t.Any]", t.cast(t.Callable[[], "InteractiveShell"], get_ipython)().config)
  except (ImportError, AttributeError): return False

_dockerenv, _cgroup = Path("/.dockerenv"), Path("/proc/self/cgroup")

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
  def compose_two(f1: AnyCallable, f2: AnyCallable) -> AnyCallable: return lambda *args, **kwargs: f1(f2(*args, **kwargs))
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

@apply(bool)
@suppress(FileNotFoundError)
def _text_in_file(text: str, filename: Path) -> bool:
  return any(text in line for line in filename.open())

def in_docker() -> bool:
  """Is this current environment running in docker?

  ```python
  type(in_docker())
  ```
  """
  return _dockerenv.exists() or _text_in_file("docker", _cgroup)

T, K = t.TypeVar("T"), t.TypeVar("K")

def resolve_filepath(path: str, ctx: str | None = None) -> str:
  """Resolve a file path to an absolute path, expand user and environment variables."""
  try: return resolve_user_filepath(path, ctx)
  except FileNotFoundError: return path

def validate_is_path(maybe_path: str) -> bool: return os.path.exists(os.path.dirname(resolve_filepath(maybe_path)))

def generate_context(framework_name: str) -> _ModelContext:
  framework_versions = {"transformers": pkg.get_pkg_version("transformers")}
  if openllm.utils.is_torch_available(): framework_versions["torch"] = pkg.get_pkg_version("torch")
  if openllm.utils.is_tf_available():
    from bentoml._internal.frameworks.utils.tensorflow import get_tf_version
    framework_versions["tensorflow"] = get_tf_version()
  if openllm.utils.is_flax_available(): framework_versions.update({"flax": pkg.get_pkg_version("flax"), "jax": pkg.get_pkg_version("jax"), "jaxlib": pkg.get_pkg_version("jaxlib")})
  return _ModelContext(framework_name=framework_name, framework_versions=framework_versions)

def generate_labels(llm: openllm.LLM[t.Any, t.Any]) -> dict[str, t.Any]: return {"runtime": llm.runtime, "framework": "openllm", "model_name": llm.config["model_name"], "architecture": llm.config["architecture"], "serialisation_format": llm._serialisation_format}

_TOKENIZER_PREFIX = "_tokenizer_"

def normalize_attrs_to_model_tokenizer_pair(**attrs: t.Any) -> tuple[dict[str, t.Any], dict[str, t.Any]]:
  """Normalize the given attrs to a model and tokenizer kwargs accordingly."""
  tokenizer_attrs = {k[len(_TOKENIZER_PREFIX):]: v for k, v in attrs.items() if k.startswith(_TOKENIZER_PREFIX)}
  for k in tuple(attrs.keys()):
    if k.startswith(_TOKENIZER_PREFIX): del attrs[k]
  return attrs, tokenizer_attrs

def infer_auto_class(implementation: LiteralRuntime) -> type[openllm.AutoLLM | openllm.AutoTFLLM | openllm.AutoFlaxLLM | openllm.AutoVLLM]:
  import openllm
  if implementation == "tf": return openllm.AutoTFLLM
  elif implementation == "flax": return openllm.AutoFlaxLLM
  elif implementation == "pt": return openllm.AutoLLM
  elif implementation == "vllm": return openllm.AutoVLLM
  else: raise RuntimeError(f"Unknown implementation: {implementation} (supported: 'pt', 'flax', 'tf', 'vllm')")

# NOTE: The set marks contains a set of modules name
# that are available above and are whitelisted
# to be included in the extra_objects map.
_whitelist_modules = {"pkg"}

# XXX: define all classes, functions import above this line
# since _extras will be the locals() import from this file.
_extras: dict[str, t.Any] = {k: v for k, v in locals().items() if k in _whitelist_modules or (not isinstance(v, types.ModuleType) and not k.startswith("_"))}
_extras["__openllm_migration__"] = {"ModelEnv": "EnvVarMixin"}
_import_structure: dict[str, list[str]] = {
  "analytics": [], "codegen": [], "dantic": [], "dummy_flax_objects": [], "dummy_pt_objects": [], "dummy_tf_objects": [], "dummy_vllm_objects": [], "representation": ["ReprMixin"], "lazy": ["LazyModule"],
  "import_utils": ["OPTIONAL_DEPENDENCIES", "ENV_VARS_TRUE_VALUES", "DummyMetaclass", "EnvVarMixin", "require_backends",
                  "is_cpm_kernels_available", "is_einops_available", "is_flax_available", "is_tf_available", "is_vllm_available", "is_torch_available", "is_bitsandbytes_available", "is_peft_available", "is_datasets_available",
                  "is_transformers_supports_kbit", "is_transformers_supports_agent", "is_jupyter_available", "is_jupytext_available", "is_notebook_available", "is_triton_available", "is_autogptq_available", "is_sentencepiece_available",
                  "is_xformers_available", "is_fairscale_available", "is_ctransformers_available"]}

if t.TYPE_CHECKING:
  # NOTE: The following exports useful utils from bentoml
  from . import (
    analytics as analytics,
    codegen as codegen,
    dantic as dantic,
    dummy_flax_objects as dummy_flax_objects,
    dummy_pt_objects as dummy_pt_objects,
    dummy_tf_objects as dummy_tf_objects,
    dummy_vllm_objects as dummy_vllm_objects,
  )
  from .import_utils import (
    ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES,
    OPTIONAL_DEPENDENCIES as OPTIONAL_DEPENDENCIES,
    DummyMetaclass as DummyMetaclass,
    EnvVarMixin as EnvVarMixin,
    is_autogptq_available as is_autogptq_available,
    is_bitsandbytes_available as is_bitsandbytes_available,
    is_cpm_kernels_available as is_cpm_kernels_available,
    is_datasets_available as is_datasets_available,
    is_einops_available as is_einops_available,
    is_fairscale_available as is_fairscale_available,
    is_flax_available as is_flax_available,
    is_jupyter_available as is_jupyter_available,
    is_jupytext_available as is_jupytext_available,
    is_notebook_available as is_notebook_available,
    is_peft_available as is_peft_available,
    is_sentencepiece_available as is_sentencepiece_available,
    is_tf_available as is_tf_available,
    is_torch_available as is_torch_available,
    is_transformers_supports_agent as is_transformers_supports_agent,
    is_transformers_supports_kbit as is_transformers_supports_kbit,
    is_triton_available as is_triton_available,
    is_vllm_available as is_vllm_available,
    is_xformers_available as is_xformers_available,
    is_ctransformers_available as is_ctransformers_available,
    require_backends as require_backends,
  )
  from .representation import ReprMixin as ReprMixin

__lazy = LazyModule(__name__, globals()["__file__"], _import_structure, extra_objects=_extras)
__all__ = __lazy.__all__
__dir__ = __lazy.__dir__
__getattr__ = __lazy.__getattr__
