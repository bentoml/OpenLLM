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

"""
Some imports utils are vendorred from transformers/utils/import_utils.py for performance reasons.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import logging
import os
import types
import typing as t
from abc import ABCMeta
from collections import OrderedDict

import attr
import inflection
from packaging import version

if t.TYPE_CHECKING:
    BackendOrderredDict = OrderedDict[str, tuple[t.Callable[[], bool], str]]
else:
    BackendOrderredDict = OrderedDict

logger = logging.getLogger(__name__)

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
_einops_available = _is_package_available("einops")
_cpm_kernel_available = _is_package_available("cpm_kernels")
_bitsandbytes_available = _is_package_available("bitsandbytes")


def is_einops_available():
    return _einops_available


def is_cpm_kernels_available():
    return _cpm_kernel_available


def is_bitsandbytes_available():
    return _bitsandbytes_available


def is_torch_available():
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


def is_tf_available():
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
                for pkg in candidates:
                    try:
                        _tf_version = importlib.metadata.version(pkg)
                        break
                    except importlib.metadata.PackageNotFoundError:
                        pass
                _tf_available = _tf_version is not None
            if _tf_available:
                if _tf_version and version.parse(_tf_version) < version.parse("2"):
                    logger.info(f"TensorFlow found but with version {_tf_version}. OpenLLM only supports TF 2.x")
                    _tf_available = False
        else:
            logger.info("Disabling Tensorflow because USE_TORCH is set")
            _tf_available = False
    return _tf_available


def is_flax_available():
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

CPM_KERNELS_IMPORT_ERROR = """{0} requires the cpm_kernels library but it was not found in your environment.
You can install it with pip: `pip install cpm_kernels`. Please note that you may need to restart your
runtime after installation.
"""

EINOPS_IMPORT_ERROR = """{0} requires the einops library but it was not found in your environment.
You can install it with pip: `pip install einops`. Please note that you may need to restart
your runtime after installation.
"""

BACKENDS_MAPPING = BackendOrderredDict(
    [
        ("flax", (is_flax_available, FLAX_IMPORT_ERROR)),
        ("tf", (is_tf_available, TENSORFLOW_IMPORT_ERROR)),
        ("torch", (is_torch_available, PYTORCH_IMPORT_ERROR)),
        ("cpm_kernels", (is_cpm_kernels_available, CPM_KERNELS_IMPORT_ERROR)),
        ("einops", (is_einops_available, EINOPS_IMPORT_ERROR)),
    ]
)


class DummyMetaclass(ABCMeta):
    """Metaclass for dummy object. It will raises ImportError
    generated by ``require_backends`` if users try to access attributes from given class
    """

    _backends: t.List[str]

    def __getattribute__(cls, key: str) -> t.Any:
        if key.startswith("_"):
            return super().__getattribute__(key)
        require_backends(cls, cls._backends)


def require_backends(o: t.Any, backends: t.MutableSequence[str]):
    if not isinstance(backends, (list, tuple)):
        backends = list(backends)

    name = o.__name__ if hasattr(o, "__name__") else o.__class__.__name__

    # Raise an error for users who might not realize that classes without "TF" are torch-only
    if "torch" in backends and "tf" not in backends and not is_torch_available() and is_tf_available():
        raise ImportError(PYTORCH_IMPORT_ERROR_WITH_TF.format(name))

    # Raise the inverse error for PyTorch users trying to load TF classes
    if "tf" in backends and "torch" not in backends and is_torch_available() and not is_tf_available():
        raise ImportError(TF_IMPORT_ERROR_WITH_PYTORCH.format(name))

    checks = (BACKENDS_MAPPING[backend] for backend in backends)
    failed = [msg.format(name) for available, msg in checks if not available()]
    if failed:
        raise ImportError("".join(failed))


@attr.define
class ModelEnv:
    model_name: str = attr.field(converter=inflection.underscore)

    module: types.ModuleType = attr.field(
        init=False,
        default=attr.Factory(
            lambda self: importlib.import_module(f".{self.model_name}", "openllm.models"),
            takes_self=True,
        ),
    )

    start_docstring: str = attr.field(
        init=False,
        default=attr.Factory(
            lambda self: getattr(
                self.module,
                f"START_{self.model_name.upper()}_COMMAND_DOCSTRING",
                f"(No docstring available for {self.model_name})",
            ),
            takes_self=True,
        ),
    )

    @property
    def framework(self) -> str:
        return f"OPENLLM_{self.model_name.upper()}_FRAMEWORK"

    @property
    def model_config(self) -> str:
        return f"OPENLLM_{self.model_name.upper()}_CONFIG"

    @property
    def model_id(self) -> str:
        return f"OPENLLM_{self.model_name.upper()}_MODEL_ID"

    @property
    def bettertransformer(self) -> str:
        return f"OPENLLM_{self.model_name.upper()}_BETTERTRANSFORMER"

    @property
    def quantize(self) -> str:
        return f"OPENLLM_{self.model_name.upper()}_QUANTIZE"

    def get_bettertransformer_env(self, default: t.Any | None = None) -> str:
        return str(os.environ.get(self.bettertransformer, str(default)).upper() in ENV_VARS_TRUE_VALUES)

    def get_quantize_env(self, default: t.Any | None = None) -> str | None:
        val = os.environ.get(self.quantize, default)
        if val is not None and val not in {"int8", "int4", "gptq"}:
            raise ValueError(f"Invalid quantization {val}, must be one of 'int8', 'int4', 'gptq'")
        return val

    def get_framework_env(self) -> t.Literal["pt", "flax", "tf"]:
        envvar = os.environ.get(self.framework, "pt")
        if envvar not in ("pt", "tf", "flax"):
            raise ValueError(f"Invalid framework implementation {envvar}, must be one of 'pt', 'tf', 'flax'")
        return envvar
