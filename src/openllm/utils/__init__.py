# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utilities function for OpenLLM.

User can import these function for convenience, but
we won't ensure backward compatibility for these functions. So use with caution.
"""
from __future__ import annotations
import contextlib
import functools
import logging
import logging.config
import os
import sys
import types
import typing as t
from pathlib import Path

from bentoml._internal.configuration import DEBUG_ENV_VAR as _DEBUG_ENV_VAR
from bentoml._internal.configuration import GRPC_DEBUG_ENV_VAR as _GRPC_DEBUG_ENV_VAR
from bentoml._internal.configuration import get_debug_mode
from bentoml._internal.configuration import get_quiet_mode
from bentoml._internal.configuration import set_quiet_mode
from bentoml._internal.log import configure_server_logging
from bentoml._internal.models.model import ModelContext as _ModelContext
from bentoml._internal.types import LazyType
from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils import bentoml_cattr
from bentoml._internal.utils import cached_contextmanager
from bentoml._internal.utils import copy_file_to_fs_folder
from bentoml._internal.utils import first_not_none
from bentoml._internal.utils import pkg
from bentoml._internal.utils import reserve_free_port
from bentoml._internal.utils import resolve_user_filepath
from bentoml._internal.utils import validate_or_create_dir

from .lazy import LazyModule


logger = logging.getLogger(__name__)

try:
    from typing import GenericAlias as _TypingGenericAlias  # type: ignore
except ImportError:
    # python < 3.9 does not have GenericAlias (list[int], tuple[str, ...] and so on)
    _TypingGenericAlias = ()  # type: ignore

if sys.version_info < (3, 10): _WithArgsTypes = (_TypingGenericAlias,)
else:
    #  _GenericAlias is the actual GenericAlias implementation
    _WithArgsTypes: t.Any = (t._GenericAlias, types.GenericAlias, types.UnionType)  # type: ignore

# NOTE: We need to do this so that overload can register
# correct overloads to typing registry
if sys.version_info[:2] >= (3, 11):
    from typing import overload as _overload
else:
    from typing_extensions import overload as _overload

if t.TYPE_CHECKING:
    import openllm

    from .._types import AnyCallable
    from .._types import DictStrAny
    from .._types import LiteralRuntime
    from .._types import P


def set_debug_mode(enabled: bool) -> None:
    # monkeypatch bentoml._internal.configuration.set_debug_mode to remove unused logs
    os.environ[_DEBUG_ENV_VAR] = str(enabled)
    os.environ[_GRPC_DEBUG_ENV_VAR] = "DEBUG" if enabled else "ERROR"


def lenient_issubclass(cls: t.Any, class_or_tuple: type[t.Any] | tuple[type[t.Any], ...] | None) -> bool:
    try: return isinstance(cls, type) and issubclass(cls, class_or_tuple)  # type: ignore[arg-type]
    except TypeError:
        if isinstance(cls, _WithArgsTypes): return False
        raise


def available_devices() -> tuple[str, ...]:
    """Return available GPU under system. Currently only supports NVIDIA GPUs."""
    from .._strategies import NvidiaGpuResource
    return tuple(NvidiaGpuResource.from_system())


@functools.lru_cache(maxsize=1)
def device_count() -> int: return len(available_devices())

# equivocal setattr to save one lookup per assignment
_object_setattr = object.__setattr__

def non_intrusive_setattr(obj: t.Any, name: str, value: t.Any) -> None:
    """This makes sure that we don't overwrite any existing attributes on the object."""
    _setattr = functools.partial(setattr, obj) if isinstance(obj, type) else _object_setattr.__get__(obj)
    if not hasattr(obj, name): _setattr(name, value)

def field_env_key(model_name: str, key: str, suffix: str | t.Literal[""] | None = None) -> str: return "_".join(filter(None, map(str.upper, ["OPENLLM", model_name, suffix.strip("_") if suffix else "", key])))

# Special debug flag controled via OPENLLMDEVDEBUG
DEBUG = sys.flags.dev_mode or (not sys.flags.ignore_environment and bool(os.environ.get("OPENLLMDEVDEBUG")))
# MYPY is like t.TYPE_CHECKING, but reserved for Mypy plugins
MYPY = False
SHOW_CODEGEN = DEBUG and int(os.environ.get("OPENLLMDEVDEBUG", str(0))) > 3


class _ExceptionFilter(logging.Filter):
    def __init__(self, exclude_exceptions: list[type[Exception]] | None = None, **kwargs: t.Any):
        from circus.exc import ConflictError
        if exclude_exceptions is None: exclude_exceptions = [ConflictError]
        else: exclude_exceptions.append(ConflictError)
        super(_ExceptionFilter, self).__init__(**kwargs)
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

_LOGGING_CONFIG: DictStrAny = {
    "version": 1,
    "disable_existing_loggers": True,
    "filters": {"excfilter": {"()": _ExceptionFilter}, "infofilter": {"()": InfoFilter}},
    "handlers": {
        "bentomlhandler": {
            "class": "logging.StreamHandler",
            "filters": ["excfilter"],
            "stream": "ext://sys.stdout",
        },
        "defaulthandler": {
            "class": "logging.StreamHandler",
            "level": logging.WARNING,
        },
    },
    "loggers": {
        "bentoml": {
            "handlers": ["bentomlhandler", "defaulthandler"],
            "level": logging.INFO,
            "propagate": False,
        },
        "openllm": {
            "handlers": ["bentomlhandler", "defaulthandler"],
            "level": logging.INFO,
            "propagate": False,
        },
    },
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
        if "IPKernelApp" not in get_ipython().config: return False
    except ImportError: return False
    except AttributeError: return False
    return True


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

    def compose_two(f1: AnyCallable, f2: t.Callable[P, t.Any]) -> t.Any:
        def _(*args: P.args, **kwargs: P.kwargs) -> t.Any: return f1(f2(*args, **kwargs))
        return _

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
    def wrap(func: t.Callable[P, t.Any]) -> t.Any: return functools.wraps(func)(compose(transform, func))
    return wrap


@apply(bool)
@suppress(FileNotFoundError)
def _text_in_file(text: str, filename: Path) -> bool: return any(text in line for line in filename.open())


def in_docker() -> bool:
    """Is this current environment running in docker?

    ```python
    type(in_docker())
    ```
    """
    return _dockerenv.exists() or _text_in_file("docker", _cgroup)


T = t.TypeVar("T")
K = t.TypeVar("K")


def resolve_filepath(path: str) -> str:
    """Resolve a file path to an absolute path, expand user and environment variables."""
    try: return resolve_user_filepath(path, None)
    except FileNotFoundError: return path

def validate_is_path(maybe_path: str) -> bool: return os.path.exists(os.path.dirname(resolve_filepath(maybe_path)))

def generate_context(framework_name: str) -> _ModelContext:
    from .import_utils import is_flax_available
    from .import_utils import is_tf_available
    from .import_utils import is_torch_available

    framework_versions = {"transformers": pkg.get_pkg_version("transformers")}
    if is_torch_available(): framework_versions["torch"] = pkg.get_pkg_version("torch")
    if is_tf_available():
        from bentoml._internal.frameworks.utils.tensorflow import get_tf_version
        framework_versions["tensorflow"] = get_tf_version()
    if is_flax_available(): framework_versions.update({"flax": pkg.get_pkg_version("flax"), "jax": pkg.get_pkg_version("jax"), "jaxlib": pkg.get_pkg_version("jaxlib")})
    return _ModelContext(framework_name=framework_name, framework_versions=framework_versions)

def generate_labels(llm: openllm.LLM[t.Any, t.Any]) -> DictStrAny:
    return {
        "runtime": llm.runtime,
        "framework": "openllm",
        "model_name": llm.config["model_name"],
        "architecture": llm.config["architecture"],
        "serialisation_format": llm._serialisation_format,
    }

_TOKENIZER_PREFIX = "_tokenizer_"
def normalize_attrs_to_model_tokenizer_pair(**attrs: t.Any) -> tuple[DictStrAny, DictStrAny]:
    """Normalize the given attrs to a model and tokenizer kwargs accordingly."""
    tokenizer_attrs = {k[len(_TOKENIZER_PREFIX) :]: v for k, v in attrs.items() if k.startswith(_TOKENIZER_PREFIX)}
    for k in tuple(attrs.keys()):
        if k.startswith(_TOKENIZER_PREFIX): del attrs[k]
    return attrs, tokenizer_attrs

@_overload
def infer_auto_class(implementation: t.Literal["pt"]) -> type[openllm.AutoLLM]: ...
@_overload
def infer_auto_class(implementation: t.Literal["tf"]) -> type[openllm.AutoTFLLM]: ...
@_overload
def infer_auto_class(implementation: t.Literal["flax"]) -> type[openllm.AutoFlaxLLM]: ...
@_overload
def infer_auto_class(implementation: t.Literal["vllm"]) -> type[openllm.AutoVLLM]: ...
def infer_auto_class(implementation: LiteralRuntime) -> type[openllm.AutoLLM] | type[openllm.AutoTFLLM] | type[openllm.AutoFlaxLLM] | type[openllm.AutoVLLM]:
    if implementation == "tf":
        from openllm import AutoTFLLM
        return AutoTFLLM
    elif implementation == "flax":
        from openllm import AutoFlaxLLM
        return AutoFlaxLLM
    elif implementation == "pt":
        from openllm import AutoLLM
        return AutoLLM
    elif implementation == "vllm":
        from openllm import AutoVLLM
        return AutoVLLM
    else: raise RuntimeError(f"Unknown implementation: {implementation} (supported: 'pt', 'flax', 'tf', 'vllm')")


# NOTE: The set marks contains a set of modules name
# that are available above and are whitelisted
# to be included in the extra_objects map.
_whitelist_modules = {"pkg"}

# XXX: define all classes, functions import above this line
# since _extras will be the locals() import from this file.
_extras: dict[str, t.Any] = {
    k: v
    for k, v in locals().items()
    if k in _whitelist_modules or (not isinstance(v, types.ModuleType) and not k.startswith("_"))
}

_extras["__openllm_migration__"] = {"ModelEnv": "EnvVarMixin"}

_import_structure: dict[str, list[str]] = {
    "analytics": [],
    "codegen": [],
    "dantic": [],
    "representation": ["ReprMixin"],
    "lazy": ["LazyModule"],
    "import_utils": [
        "OPTIONAL_DEPENDENCIES",
        "ENV_VARS_TRUE_VALUES",
        "DummyMetaclass",
        "EnvVarMixin",
        "requires_dependencies",
        "is_cpm_kernels_available",
        "is_einops_available",
        "is_flax_available",
        "is_tf_available",
        "is_vllm_available",
        "is_torch_available",
        "is_bitsandbytes_available",
        "is_peft_available",
        "is_datasets_available",
        "is_transformers_supports_kbit",
        "is_transformers_supports_agent",
        "is_jupyter_available",
        "is_jupytext_available",
        "is_notebook_available",
        "is_triton_available",
        "is_autogptq_available",
        "require_backends",
    ],
}

if t.TYPE_CHECKING:
    # NOTE: The following exports useful utils from bentoml
    from . import LazyLoader as LazyLoader
    from . import LazyType as LazyType
    from . import analytics as analytics
    from . import bentoml_cattr as bentoml_cattr
    from . import cached_contextmanager as cached_contextmanager
    from . import codegen as codegen
    from . import configure_logging as configure_logging
    from . import configure_server_logging as configure_server_logging
    from . import copy_file_to_fs_folder as copy_file_to_fs_folder
    from . import dantic as dantic
    from . import first_not_none as first_not_none
    from . import reserve_free_port as reserve_free_port
    from . import set_debug_mode as set_debug_mode
    from . import set_quiet_mode as set_quiet_mode
    from . import validate_is_path as validate_is_path
    from . import validate_or_create_dir as validate_or_create_dir
    from .import_utils import ENV_VARS_TRUE_VALUES as ENV_VARS_TRUE_VALUES
    from .import_utils import OPTIONAL_DEPENDENCIES as OPTIONAL_DEPENDENCIES
    from .import_utils import DummyMetaclass as DummyMetaclass
    from .import_utils import EnvVarMixin as EnvVarMixin
    from .import_utils import is_autogptq_available as is_autogptq_available
    from .import_utils import is_bitsandbytes_available as is_bitsandbytes_available
    from .import_utils import is_cpm_kernels_available as is_cpm_kernels_available
    from .import_utils import is_datasets_available as is_datasets_available
    from .import_utils import is_einops_available as is_einops_available
    from .import_utils import is_flax_available as is_flax_available
    from .import_utils import is_jupyter_available as is_jupyter_available
    from .import_utils import is_jupytext_available as is_jupytext_available
    from .import_utils import is_notebook_available as is_notebook_available
    from .import_utils import is_peft_available as is_peft_available
    from .import_utils import is_tf_available as is_tf_available
    from .import_utils import is_torch_available as is_torch_available
    from .import_utils import is_transformers_supports_agent as is_transformers_supports_agent
    from .import_utils import is_transformers_supports_kbit as is_transformers_supports_kbit
    from .import_utils import is_triton_available as is_triton_available
    from .import_utils import is_vllm_available as is_vllm_available
    from .import_utils import require_backends as require_backends
    from .import_utils import requires_dependencies as requires_dependencies
    from .representation import ReprMixin as ReprMixin
else:
    import sys

    sys.modules[__name__] = LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extras,
    )
