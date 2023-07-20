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

"""Some imports utils are vendorred from transformers/utils/import_utils.py for performance reasons."""
from __future__ import annotations
import functools
import importlib
import importlib.metadata
import importlib.util
import logging
import os
import sys
import typing as t
from abc import ABCMeta
from collections import OrderedDict

import inflection
from packaging import version

from bentoml._internal.utils import LazyLoader
from bentoml._internal.utils import pkg

from .representation import ReprMixin


# NOTE: We need to do this so that overload can register
# correct overloads to typing registry
if sys.version_info[:2] >= (3, 11):
    from typing import overload
else:
    from typing_extensions import overload

if t.TYPE_CHECKING:
    BackendOrderredDict = OrderedDict[str, tuple[t.Callable[[], bool], str]]
    from .._types import LiteralRuntime
    from .._types import P
    from .._types import T

    class _AnnotatedLazyLoader(LazyLoader, t.Generic[T]):
        DEFAULT_PROMPT_TEMPLATE: t.LiteralString | None | t.Callable[[T], t.LiteralString]
        PROMPT_MAPPING: dict[T, t.LiteralString] | None

else:
    _AnnotatedLazyLoader = LazyLoader
    BackendOrderredDict = OrderedDict

logger = logging.getLogger(__name__)

OPTIONAL_DEPENDENCIES = {
    "opt",
    "flan-t5",
    "vllm",
    "fine-tune",
    "ggml",
    "agents",
    "openai",
    "playground",
    "gptq",
}
ENV_VARS_TRUE_VALUES = {"1", "ON", "YES", "TRUE"}
ENV_VARS_TRUE_AND_AUTO_VALUES = ENV_VARS_TRUE_VALUES.union({"AUTO"})

USE_TF = os.environ.get("USE_TF", "AUTO").upper()
USE_TORCH = os.environ.get("USE_TORCH", "AUTO").upper()
USE_JAX = os.environ.get("USE_FLAX", "AUTO").upper()

FORCE_TF_AVAILABLE = os.environ.get("FORCE_TF_AVAILABLE", "AUTO").upper()


def _is_package_available(package: str) -> bool:
    _package_available = importlib.util.find_spec(package) is not None
    if _package_available:
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            _package_available = False
    return _package_available


_torch_available = importlib.util.find_spec("torch") is not None
_tf_available = importlib.util.find_spec("tensorflow") is not None
_flax_available = importlib.util.find_spec("jax") is not None and importlib.util.find_spec("flax") is not None
_vllm_available = importlib.util.find_spec("vllm") is not None

_peft_available = _is_package_available("peft")
_einops_available = _is_package_available("einops")
_cpm_kernel_available = _is_package_available("cpm_kernels")
_bitsandbytes_available = _is_package_available("bitsandbytes")
_datasets_available = _is_package_available("datasets")
_triton_available = _is_package_available("triton")
_jupyter_available = _is_package_available("jupyter")
_jupytext_available = _is_package_available("jupytext")
_notebook_available = _is_package_available("notebook")
_autogptq_available = _is_package_available("auto_gptq")


def is_transformers_supports_kbit() -> bool:
    return pkg.pkg_version_info("transformers")[:2] >= (4, 30)


def is_transformers_supports_agent() -> bool:
    return pkg.pkg_version_info("transformers")[:2] >= (4, 29)


def is_jupyter_available() -> bool:
    return _jupyter_available


def is_jupytext_available() -> bool:
    return _jupytext_available


def is_notebook_available() -> bool:
    return _notebook_available


def is_triton_available() -> bool:
    return _triton_available


def is_datasets_available() -> bool:
    return _datasets_available


def is_peft_available() -> bool:
    return _peft_available


def is_einops_available() -> bool:
    return _einops_available


def is_cpm_kernels_available() -> bool:
    return _cpm_kernel_available


def is_bitsandbytes_available() -> bool:
    return _bitsandbytes_available


def is_autogptq_available() -> bool:
    return _autogptq_available


def is_vllm_available() -> bool:
    return _vllm_available


def is_torch_available() -> bool:
    global _torch_available
    if USE_TORCH in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TF not in ENV_VARS_TRUE_VALUES:
        if _torch_available:
            try:
                importlib.metadata.version("torch")
            except importlib.metadata.PackageNotFoundError:
                _torch_available = False
    else:
        logger.info("Disabling PyTorch because USE_TF is set")
        _torch_available = False
    return _torch_available


def is_tf_available() -> bool:
    global _tf_available
    if FORCE_TF_AVAILABLE in ENV_VARS_TRUE_VALUES:
        _tf_available = True
    else:
        _tf_version = None
        if USE_TF in ENV_VARS_TRUE_AND_AUTO_VALUES and USE_TORCH not in ENV_VARS_TRUE_VALUES:
            if _tf_available:
                candidates = (
                    "tensorflow",
                    "tensorflow-cpu",
                    "tensorflow-gpu",
                    "tf-nightly",
                    "tf-nightly-cpu",
                    "tf-nightly-gpu",
                    "intel-tensorflow",
                    "intel-tensorflow-avx512",
                    "tensorflow-rocm",
                    "tensorflow-macos",
                    "tensorflow-aarch64",
                )
                _tf_version = None
                # For the metadata, we have to look for both tensorflow and tensorflow-cpu
                for _pkg in candidates:
                    try:
                        _tf_version = importlib.metadata.version(_pkg)
                        break
                    except importlib.metadata.PackageNotFoundError:
                        pass
                _tf_available = _tf_version is not None
            if _tf_available:
                if _tf_version and version.parse(_tf_version) < version.parse("2"):
                    logger.info("TensorFlow found but with version %s. OpenLLM only supports TF 2.x", _tf_version)
                    _tf_available = False
        else:
            logger.info("Disabling Tensorflow because USE_TORCH is set")
            _tf_available = False
    return _tf_available


def is_flax_available() -> bool:
    global _flax_available
    if USE_JAX in ENV_VARS_TRUE_AND_AUTO_VALUES:
        if _flax_available:
            try:
                importlib.metadata.version("jax")
                importlib.metadata.version("flax")
            except importlib.metadata.PackageNotFoundError:
                _flax_available = False
    else:
        _flax_available = False
    return _flax_available


def requires_dependencies(
    package: str | list[str], *, extra: str | list[str] | None = None
) -> t.Callable[[t.Callable[P, t.Any]], t.Callable[P, t.Any]]:
    import openllm.utils

    if isinstance(package, str):
        package = [package]
    if isinstance(extra, str):
        extra = [extra]

    def decorator(func: t.Callable[P, t.Any]):
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> t.Any:
            for p in package:
                cached_check: t.Callable[[], bool] | None = getattr(openllm.utils, f"is_{p}_available", None)
                if not ((cached_check is not None and cached_check()) or _is_package_available(p)):
                    raise ImportError(
                        f"{func.__name__} requires '{p}' to be available locally (Currently missing)."
                        f"Make sure to have {p} to be installed: 'pip install \"{p if not extra else 'openllm['+', '.join(extra)+']'}\"'"
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator


VLLM_IMPORT_ERROR_WITH_PYTORCH = """\
{0} requires the vLLM library but it was not found in your environment.
However, we were able to find a PyTorch installation. PyTorch classes do not begin
with "VLLM", but are otherwise identically named to our PyTorch classes.
If you want to use PyTorch, please use those classes instead!

If you really do want to use vLLM, please follow the instructions on the
installation page https://github.com/vllm-project/vllm that match your environment.
"""

VLLM_IMPORT_ERROR_WITH_TF = """\
{0} requires the vLLM library but it was not found in your environment.
However, we were able to find a TensorFlow installation. TensorFlow classes begin
with "TF", but are otherwise identically named to the PyTorch classes. This
means that the TF equivalent of the class you tried to import would be "TF{0}".
If you want to use TensorFlow, please use TF classes instead!

If you really do want to use vLLM, please follow the instructions on the
installation page https://github.com/vllm-project/vllm that match your environment.
"""

VLLM_IMPORT_ERROR_WITH_FLAX = """\
{0} requires the vLLM library but it was not found in your environment.
However, we were able to find a Flax installation. Flax classes begin
with "Flax", but are otherwise identically named to the PyTorch classes. This
means that the Flax equivalent of the class you tried to import would be "Flax{0}".
If you want to use Flax, please use Flax classes instead!

If you really do want to use vLLM, please follow the instructions on the
installation page https://github.com/vllm-project/vllm that match your environment.
"""

PYTORCH_IMPORT_ERROR_WITH_TF = """\
{0} requires the PyTorch library but it was not found in your environment.
However, we were able to find a TensorFlow installation. TensorFlow classes begin
with "TF", but are otherwise identically named to the PyTorch classes. This
means that the TF equivalent of the class you tried to import would be "TF{0}".
If you want to use TensorFlow, please use TF classes instead!

If you really do want to use PyTorch please go to
https://pytorch.org/get-started/locally/ and follow the instructions that
match your environment.
"""

TF_IMPORT_ERROR_WITH_PYTORCH = """\
{0} requires the TensorFlow library but it was not found in your environment.
However, we were able to find a PyTorch installation. PyTorch classes do not begin
with "TF", but are otherwise identically named to our TF classes.
If you want to use PyTorch, please use those classes instead!

If you really do want to use TensorFlow, please follow the instructions on the
installation page https://www.tensorflow.org/install that match your environment.
"""

TENSORFLOW_IMPORT_ERROR = """{0} requires the TensorFlow library but it was not found in your environment.
Checkout the instructions on the installation page: https://www.tensorflow.org/install and follow the
ones that match your environment. Please note that you may need to restart your runtime after installation.
"""


FLAX_IMPORT_ERROR = """{0} requires the FLAX library but it was not found in your environment.
Checkout the instructions on the installation page: https://github.com/google/flax and follow the
ones that match your environment. Please note that you may need to restart your runtime after installation.
"""

PYTORCH_IMPORT_ERROR = """{0} requires the PyTorch library but it was not found in your environment.
Checkout the instructions on the installation page: https://pytorch.org/get-started/locally/ and follow the
ones that match your environment. Please note that you may need to restart your runtime after installation.
"""

VLLM_IMPORT_ERROR = """{0} requires the vLLM library but it was not found in your environment.
Checkout the instructions on the installation page: https://github.com/vllm-project/vllm
ones that match your environment. Please note that you may need to restart your runtime after installation.
"""

CPM_KERNELS_IMPORT_ERROR = """{0} requires the cpm_kernels library but it was not found in your environment.
You can install it with pip: `pip install cpm_kernels`. Please note that you may need to restart your
runtime after installation.
"""

EINOPS_IMPORT_ERROR = """{0} requires the einops library but it was not found in your environment.
You can install it with pip: `pip install einops`. Please note that you may need to restart
your runtime after installation.
"""

TRITON_IMPORT_ERROR = """{0} requires the triton library but it was not found in your environment.
You can install it with pip: 'pip install \"git+https://github.com/openai/triton.git#egg=triton&subdirectory=python\"'.
Please note that you may need to restart your runtime after installation.
"""

DATASETS_IMPORT_ERROR = """{0} requires the datasets library but it was not found in your environment.
You can install it with pip: `pip install datasets`. Please note that you may need to restart
your runtime after installation.
"""

PEFT_IMPORT_ERROR = """{0} requires the peft library but it was not found in your environment.
You can install it with pip: `pip install peft`. Please note that you may need to restart
your runtime after installation.
"""

BITSANDBYTES_IMPORT_ERROR = """{0} requires the bitsandbytes library but it was not found in your environment.
You can install it with pip: `pip install bitsandbytes`. Please note that you may need to restart
your runtime after installation.
"""

AUTOGPTQ_IMPORT_ERROR = """{0} requires the auto-gptq library but it was not found in your environment.
You can install it with pip: `pip install auto-gptq`. Please note that you may need to restart
your runtime after installation.
"""

BACKENDS_MAPPING = BackendOrderredDict(
    [
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("vllm", (is_vllm_available, VLLM_IMPORT_ERROR)),
        ("cpm_kernels", (is_cpm_kernels_available, CPM_KERNELS_IMPORT_ERROR)),
        ("einops", (is_einops_available, EINOPS_IMPORT_ERROR)),
        ("triton", (is_triton_available, TRITON_IMPORT_ERROR)),
        ("datasets", (is_datasets_available, DATASETS_IMPORT_ERROR)),
        ("peft", (is_peft_available, PEFT_IMPORT_ERROR)),
        ("bitsandbytes", (is_bitsandbytes_available, BITSANDBYTES_IMPORT_ERROR)),
        ("auto-gptq", (is_autogptq_available, AUTOGPTQ_IMPORT_ERROR)),
    ]
)


class DummyMetaclass(ABCMeta):
    """Metaclass for dummy object.

    It will raises ImportError generated by ``require_backends`` if users try to access attributes from given class.
    """

    _backends: t.List[str]

    def __getattribute__(cls, key: str) -> t.Any:
        if key.startswith("_"):
            return super().__getattribute__(key)
        require_backends(cls, cls._backends)


def require_backends(o: t.Any, backends: t.MutableSequence[str]) -> None:
    if not isinstance(backends, (list, tuple)):
        backends = list(backends)

    name = o.__name__ if hasattr(o, "__name__") else o.__class__.__name__

    # Raise an error for users who might not realize that classes without "TF" are torch-only
    if "torch" in backends and "tf" not in backends and not is_torch_available() and is_tf_available():
        raise ImportError(PYTORCH_IMPORT_ERROR_WITH_TF.format(name))

    # Raise the inverse error for PyTorch users trying to load TF classes
    if "tf" in backends and "torch" not in backends and is_torch_available() and not is_tf_available():
        raise ImportError(TF_IMPORT_ERROR_WITH_PYTORCH.format(name))

    if "vllm" in backends:
        if "torch" not in backends and is_torch_available() and not is_vllm_available():
            raise ImportError(VLLM_IMPORT_ERROR_WITH_PYTORCH.format(name))

        if "tf" not in backends and is_tf_available() and not is_vllm_available():
            raise ImportError(VLLM_IMPORT_ERROR_WITH_TF.format(name))

        if "flax" not in backends and is_flax_available() and not is_vllm_available():
            raise ImportError(VLLM_IMPORT_ERROR_WITH_FLAX.format(name))

    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


class EnvVarMixin(ReprMixin):
    model_name: str

    @property
    def __repr_keys__(self) -> set[str]:
        return {"config", "model_id", "quantize", "framework", "bettertransformer", "runtime"}

    if t.TYPE_CHECKING:
        config: str
        model_id: str
        quantize: str
        framework: str
        bettertransformer: str
        runtime: t.Literal["ggml", "transformers"]

        framework_value: LiteralRuntime
        quantize_value: str | None
        bettertransformer_value: str | None
        runtime_value: t.Literal["ggml", "transformers"]

    # fmt: off
    @overload
    def __getitem__(self, item: t.Literal["config"]) -> str: ...
    @overload
    def __getitem__(self, item: t.Literal["model_id"]) -> str: ...
    @overload
    def __getitem__(self, item: t.Literal["quantize"]) -> str: ...
    @overload
    def __getitem__(self, item: t.Literal["framework"]) -> str: ...
    @overload
    def __getitem__(self, item: t.Literal["bettertransformer"]) -> str: ...
    @overload
    def __getitem__(self, item: t.Literal["runtime"]) -> str: ...
    @overload
    def __getitem__(self, item: t.Literal["framework_value"]) -> LiteralRuntime: ...
    @overload
    def __getitem__(self, item: t.Literal["quantize_value"]) -> str | None: ...
    @overload
    def __getitem__(self, item: t.Literal["model_id_value"]) -> str | None: ...
    @overload
    def __getitem__(self, item: t.Literal["bettertransformer_value"]) -> str | None: ...
    @overload
    def __getitem__(self, item: t.Literal["runtime_value"]) -> t.Literal["ggml", "transformers"]: ...
    # fmt: on

    def __getitem__(self, item: str | t.Any) -> t.Any:
        if hasattr(self, item):
            return getattr(self, item)
        raise KeyError(f"Key {item} not found in {self}")

    def __new__(
        cls,
        model_name: str,
        implementation: LiteralRuntime = "pt",
        bettertransformer: bool | None = None,
        quantize: t.LiteralString | None = None,
        runtime: t.Literal["ggml", "transformers"] = "transformers",
    ) -> t.Self:
        from . import codegen
        from .._configuration import field_env_key

        model_name = inflection.underscore(model_name)

        res = super().__new__(cls)
        res.model_name = model_name

        # gen properties env key
        attributes = {"config", "model_id", "quantize", "framework", "bettertransformer", "runtime"}
        for att in attributes:
            setattr(res, att, field_env_key(model_name, att.upper()))

        # gen properties env value
        attributes_with_values = {
            "framework": (str, implementation),
            "quantize": (str, quantize),
            "bettertransformer": (bool, bettertransformer),
            "model_id": (str, None),
            "runtime": (str, runtime),
        }
        globs: dict[str, t.Any] = {
            "__bool_vars_value": ENV_VARS_TRUE_VALUES,
            "__env_get": os.getenv,
            "self": res,
        }

        for attribute, (default_type, default_value) in attributes_with_values.items():
            lines: list[str] = []
            if default_type is bool:
                lines.append(
                    f"return str(__env_get(self['{attribute}'], str(__env_default)).upper() in __bool_vars_value)"
                )
            else:
                lines.append(f"return __env_get(self['{attribute}'], __env_default)")

            setattr(
                res,
                f"{attribute}_value",
                codegen.generate_function(
                    cls,
                    "_env_get_" + attribute,
                    lines,
                    ("__env_default",),
                    globs,
                )(default_value),
            )

        return res

    @property
    def start_docstring(self) -> str:
        return getattr(self.module, f"START_{self.model_name.upper()}_COMMAND_DOCSTRING")

    @property
    def module(self) -> _AnnotatedLazyLoader[t.LiteralString]:
        return _AnnotatedLazyLoader(self.model_name, globals(), f"openllm.models.{self.model_name}")
