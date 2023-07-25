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

from __future__ import annotations
import functools
import inspect
import logging
import math
import os
import sys
import types
import typing as t
import warnings

import psutil

import bentoml
from bentoml._internal.resource import get_resource
from bentoml._internal.resource import system_resources
from bentoml._internal.runner.strategy import THREAD_ENVS

from .utils import DEBUG
from .utils import LazyType
from .utils import ReprMixin


if t.TYPE_CHECKING:
    ListIntStr = list[int | str]
    class DynResource(bentoml.Resource[t.List[str]], resource_id=""):
        resource_id: t.ClassVar[str]

else:
    DynResource = bentoml.Resource[t.List[str]]
    ListIntStr = list

# NOTE: We need to do this so that overload can register
# correct overloads to typing registry
if sys.version_info[:2] >= (3, 11):
    from typing import overload
else:
    from typing_extensions import overload

logger = logging.getLogger(__name__)

def _strtoul(s: str) -> int:
    """Return -1 or positive integer sequence string starts with,."""
    if not s: return -1
    idx = 0
    for idx, c in enumerate(s):
        if not (c.isdigit() or (idx == 0 and c in "+-")): break
        if idx + 1 == len(s): idx += 1  # noqa: PLW2901
    # NOTE: idx will be set via enumerate
    return int(s[:idx]) if idx > 0 else -1


def _parse_list_with_prefix(lst: str, prefix: str) -> list[str]:
    rcs: list[str] = []
    for elem in lst.split(","):
        # Repeated id results in empty set
        if elem in rcs: return []
        # Anything other but prefix is ignored
        if not elem.startswith(prefix): break
        rcs.append(elem)
    return rcs


_STACK_LEVEL = 3

@overload
def _parse_visible_devices(default_var: str | None = ..., respect_env: t.Literal[True] = True) -> list[str] | None: ...
@overload
def _parse_visible_devices(default_var: str = ..., respect_env: t.Literal[False] = False) -> list[str]: ...
def _parse_visible_devices(default_var: str | None = None, respect_env: bool = True) -> list[str] | None:
    """CUDA_VISIBLE_DEVICES aware with default var for parsing spec."""
    if respect_env:
        spec = os.getenv("CUDA_VISIBLE_DEVICES", default_var)
        if not spec: return
    else:
        assert default_var is not None, "spec is required to be not None when parsing spec."  # noqa: S101
        spec = default_var

    if spec.startswith("GPU-"): return _parse_list_with_prefix(spec, "GPU-")
    if spec.startswith("MIG-"): return _parse_list_with_prefix(spec, "MIG-")

    # XXX: We to somehow handle cases such as '100m'
    # CUDA_VISIBLE_DEVICES uses something like strtoul
    # which makes `1gpu2,2ampere` is equivalent to `1,2`
    rc: list[int] = []
    for el in spec.split(","):
        x = _strtoul(el.strip())
        # Repeated ordinal results in empty set
        if x in rc: return []
        # Negative value aborts the sequence
        if x < 0: break
        rc.append(x)
    return [str(i) for i in rc]


def _from_system(cls: type[DynResource]) -> list[str]:
    """Shared mixin implementation for OpenLLM's NVIDIA and AMD resource implementation.

    It relies on torch.cuda implementation and in turns respect CUDA_VISIBLE_DEVICES.
    """
    visible_devices = _parse_visible_devices()
    if visible_devices is None:
        if cls.resource_id == "amd.com/gpu":
            if not psutil.LINUX:
                if DEBUG: warnings.warn("AMD GPUs is currently only supported on Linux.", stacklevel=_STACK_LEVEL)
                return []

            # ROCm does not currently have the rocm_smi wheel.
            # So we need to use the ctypes bindings directly.
            # we don't want to use CLI because parsing is a pain.
            sys.path.append("/opt/rocm/libexec/rocm_smi")
            try:
                from ctypes import byref
                from ctypes import c_uint32

                # refers to https://github.com/RadeonOpenCompute/rocm_smi_lib/blob/master/python_smi_tools/rsmiBindings.py
                from rsmiBindings import rocmsmi
                from rsmiBindings import rsmi_status_t

                device_count = c_uint32(0)
                ret = rocmsmi.rsmi_num_monitor_devices(byref(device_count))
                if ret == rsmi_status_t.RSMI_STATUS_SUCCESS: return [str(i) for i in range(device_count.value)]
                return []
            # In this case the binary is not found, returning empty list
            except (ModuleNotFoundError, ImportError): return []
            finally: sys.path.remove("/opt/rocm/libexec/rocm_smi")
        else:
            try:
                from cuda import cuda
                err, *_ = cuda.cuInit(0)
                if err != cuda.CUresult.CUDA_SUCCESS:
                    logger.warning("Failed to initialise CUDA", stacklevel=_STACK_LEVEL)
                    return []
                _, dev = cuda.cuDeviceGetCount()
                return [str(i) for i in range(dev)]
            except (ImportError, RuntimeError): return []
    return visible_devices


@overload
def _from_spec(cls: type[DynResource], spec: int) -> list[str]: ...
@overload
def _from_spec(cls: type[DynResource], spec: ListIntStr) -> list[str]: ...
@overload
def _from_spec(cls: type[DynResource], spec: str) -> list[str]: ...
def _from_spec(cls: type[DynResource], spec: t.Any) -> list[str]:
    """Shared mixin implementation for OpenLLM's NVIDIA and AMD resource implementation.

    The parser behaves similar to how PyTorch handles CUDA_VISIBLE_DEVICES. This means within
    BentoML's resource configuration, its behaviour is similar to CUDA_VISIBLE_DEVICES.
    """
    if isinstance(spec, int):
        if spec in (-1, 0): return []
        if spec < -1: raise ValueError("Spec cannot be < -1.")
        return [str(i) for i in range(spec)]
    elif isinstance(spec, str):
        if not spec: return []
        if spec.isdigit(): spec = ",".join([str(i) for i in range(_strtoul(spec))])
        return _parse_visible_devices(spec, respect_env=False)
    elif LazyType(ListIntStr).isinstance(spec): return [str(x) for x in spec]
    else: raise TypeError(f"'{cls.__name__}.from_spec' only supports parsing spec of type int, str, or list, got '{type(spec)}' instead.")


def _raw_device_uuid_nvml() -> list[str] | None:
    """Return list of device UUID as reported by NVML or None if NVML discovery/initialization failed."""
    from ctypes import CDLL
    from ctypes import byref
    from ctypes import c_int
    from ctypes import c_void_p
    from ctypes import create_string_buffer

    try: nvml_h = CDLL("libnvidia-ml.so.1")
    except Exception:
        warnings.warn("Failed to find nvidia binding", stacklevel=_STACK_LEVEL)
        return

    rc = nvml_h.nvmlInit()
    if rc != 0:
        warnings.warn("Can't initialize NVML", stacklevel=_STACK_LEVEL)
        return
    dev_count = c_int(-1)
    rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
    if rc != 0:
        warnings.warn("Failed to get available device from system.", stacklevel=_STACK_LEVEL)
        return
    uuids: list[str] = []
    for idx in range(dev_count.value):
        dev_id = c_void_p()
        rc = nvml_h.nvmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
        if rc != 0:
            warnings.warn(f"Failed to get device handle for {idx}", stacklevel=_STACK_LEVEL)
            return
        buf_len = 96
        buf = create_string_buffer(buf_len)
        rc = nvml_h.nvmlDeviceGetUUID(dev_id, buf, buf_len)
        if rc != 0:
            warnings.warn(f"Failed to get device UUID for {idx}", stacklevel=_STACK_LEVEL)
            return
        uuids.append(buf.raw.decode("ascii").strip("\0"))
    del nvml_h
    return uuids


def _validate(cls: type[DynResource], val: list[t.Any]):
    if cls.resource_id == "amd.com/gpu":
        raise RuntimeError("AMD GPU validation is not yet supported. Make sure to call 'get_resource(..., validate=False)'")
    if not all(isinstance(i, str) for i in val): raise ValueError("Input list should be all string type.")

    try:
        from cuda import cuda

        err, *_ = cuda.cuInit(0)
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Failed to initialise CUDA runtime binding.")
        # correctly parse handle
        for el in val:
            if el.startswith("GPU-") or el.startswith("MIG-"):
                uuids = _raw_device_uuid_nvml()
                if uuids is None: raise ValueError("Failed to parse available GPUs UUID")
                if el not in uuids: raise ValueError(f"Given UUID {el} is not found with available UUID (available: {uuids})")
            elif el.isdigit():
                err, _ = cuda.cuDeviceGet(int(el))
                if err != cuda.CUresult.CUDA_SUCCESS: raise ValueError(f"Failed to get device {el}")
    except (ImportError, RuntimeError):
        pass


def _make_resource_class(name: str, resource_kind: str, docstring: str) -> type[DynResource]:
    return types.new_class(
        name,
        (DynResource, ReprMixin),
        {"resource_id": resource_kind},
        lambda ns: ns.update(
            {
                "resource_id": resource_kind,
                "from_spec": classmethod(_from_spec),
                "from_system": classmethod(_from_system),
                "validate": classmethod(_validate),
                "__repr_keys__": property(lambda _: {"resource_id"}),
                "__doc__": inspect.cleandoc(docstring),
                "__module__": "openllm._strategies",
            }
        ),
    )

_TPU_RESOURCE = "cloud-tpus.google.com/v2"
_AMD_GPU_RESOURCE = "amd.com/gpu"
_NVIDIA_GPU_RESOURCE = "nvidia.com/gpu"
_CPU_RESOURCE = "cpu"

NvidiaGpuResource = _make_resource_class(
    "NvidiaGpuResource",
    _NVIDIA_GPU_RESOURCE,
    """NVIDIA GPU resource.

    This is a modified version of internal's BentoML's NvidiaGpuResource
    where it respects and parse CUDA_VISIBLE_DEVICES correctly.""",
)
AmdGpuResource = _make_resource_class(
    "AmdGpuResource",
    _AMD_GPU_RESOURCE,
    """AMD GPU resource.

    Since ROCm will respect CUDA_VISIBLE_DEVICES, the behaviour of from_spec, from_system are similar to
    ``NvidiaGpuResource``. Currently ``validate`` is not yet supported.""",
)

LiteralResourceSpec = t.Literal["cloud-tpus.google.com/v2", "amd.com/gpu", "nvidia.com/gpu", "cpu"]

# convenient mapping
def resource_spec(name: t.Literal["tpu", "amd", "nvidia", "cpu"]) ->  LiteralResourceSpec:
    if name == "tpu": return _TPU_RESOURCE
    elif name == "amd": return _AMD_GPU_RESOURCE
    elif name == "nvidia": return _NVIDIA_GPU_RESOURCE
    elif name == "cpu": return _CPU_RESOURCE
    else: raise ValueError("Unknown alias. Accepted: ['tpu', 'amd', 'nvidia', 'cpu']")
@functools.lru_cache
def available_resource_spec() -> tuple[LiteralResourceSpec, ...]:
    """This is a utility function helps to determine the available resources from given running system.

    It will first check for TPUs -> AMD GPUS -> NVIDIA GPUS -> CPUs.

    TODO: Supports TPUs
    """
    available = ()
    if len(AmdGpuResource.from_system()) > 0: available += (_AMD_GPU_RESOURCE,)
    if len(NvidiaGpuResource.from_system()) > 0: available += (_NVIDIA_GPU_RESOURCE,)
    available += (_CPU_RESOURCE,)
    return t.cast(t.Tuple[LiteralResourceSpec, ...], available)


class CascadingResourceStrategy(bentoml.Strategy, ReprMixin):
    """This is extends the default BentoML strategy where we check for NVIDIA GPU resource -> AMD GPU resource -> CPU resource.

    It also respect CUDA_VISIBLE_DEVICES for both AMD and NVIDIA GPU.
    See https://rocm.docs.amd.com/en/develop/understand/gpu_isolation.html#cuda-visible-devices
    for ROCm's support for CUDA_VISIBLE_DEVICES.

    TODO: Support CloudTPUResource
    """

    @property
    def __repr_keys__(self) -> set[str]:
        return set()

    @classmethod
    def get_worker_count(
        cls,
        runnable_class: type[bentoml.Runnable],
        resource_request: dict[str, t.Any] | None,
        workers_per_resource: int | float,
    ) -> int:
        if resource_request is None:
            resource_request = system_resources()

        def _get_gpu_count(typ: list[str] | None, kind: str):
            if typ is not None and len(typ) > 0 and kind in runnable_class.SUPPORTED_RESOURCES:
                return math.ceil(len(typ) * workers_per_resource)

        # use NVIDIA
        kind = "nvidia.com/gpu"
        count = _get_gpu_count(get_resource(resource_request, kind), kind)
        if count:
            return count

        # use AMD
        kind = "amd.com/gpu"
        count = _get_gpu_count(get_resource(resource_request, kind, validate=False), kind)
        if count:
            return count

        # use CPU
        cpus = get_resource(resource_request, "cpu")
        if cpus is not None and cpus > 0:
            if "cpu" not in runnable_class.SUPPORTED_RESOURCES:
                logger.warning(
                    "No known supported resource available for %s, falling back to using CPU.",
                    runnable_class,
                )

            if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
                if isinstance(workers_per_resource, float) and workers_per_resource < 1.0:
                    raise ValueError("Fractional CPU multi threading support is not yet supported.")
                return int(workers_per_resource)

            return math.ceil(cpus) * workers_per_resource

        # this should not be reached by user since we always read system resource as default
        raise ValueError(
            f"No known supported resource available for {runnable_class}. Please check your resource request. "
            "Leaving it blank will allow BentoML to use system resources."
        )

    @classmethod
    def get_worker_env(
        cls,
        runnable_class: type[bentoml.Runnable],
        resource_request: dict[str, t.Any] | None,
        workers_per_resource: int | float,
        worker_index: int,
    ) -> dict[str, t.Any]:
        """Get worker env for this given worker_index.

        Args:
            runnable_class: The runnable class to be run.
            resource_request: The resource request of the runnable.
            workers_per_resource: # of workers per resource.
            worker_index: The index of the worker, start from 0.
        """
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        disabled = cuda_env in ("", "-1")

        environ: dict[str, t.Any] = {}

        if resource_request is None:
            resource_request = system_resources()

        # use NVIDIA
        kind = "nvidia.com/gpu"
        typ = get_resource(resource_request, kind)
        if typ is not None and len(typ) > 0 and kind in runnable_class.SUPPORTED_RESOURCES:
            if disabled:
                logger.debug("CUDA_VISIBLE_DEVICES is disabled, %s will not be using GPU.", worker_index)
                environ["CUDA_VISIBLE_DEVICES"] = cuda_env
                return environ
            environ["CUDA_VISIBLE_DEVICES"] = cls.transpile_workers_to_cuda_envvar(
                workers_per_resource, typ, worker_index
            )
            logger.debug("Environ for worker %s: %s", worker_index, environ)
            return environ

        # use AMD
        kind = "amd.com/gpu"
        typ = get_resource(resource_request, kind, validate=False)
        if typ is not None and len(typ) > 0 and kind in runnable_class.SUPPORTED_RESOURCES:
            if disabled:
                logger.debug("CUDA_VISIBLE_DEVICES is disabled, %s will not be using GPU.", worker_index)
                environ["CUDA_VISIBLE_DEVICES"] = cuda_env
                return environ
            environ["CUDA_VISIBLE_DEVICES"] = cls.transpile_workers_to_cuda_envvar(
                workers_per_resource, typ, worker_index
            )
            logger.debug("Environ for worker %s: %s", worker_index, environ)
            return environ

        # use CPU
        cpus = get_resource(resource_request, "cpu")
        if cpus is not None and cpus > 0:
            environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu
            if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
                thread_count = math.ceil(cpus)
                for thread_env in THREAD_ENVS:
                    environ[thread_env] = os.getenv(thread_env, str(thread_count))
                logger.debug("Environ for worker %s: %s", worker_index, environ)
                return environ
            for thread_env in THREAD_ENVS:
                environ[thread_env] = os.getenv(thread_env, "1")
            return environ

        return environ

    @staticmethod
    def transpile_workers_to_cuda_envvar(workers_per_resource: float | int, gpus: list[str], worker_index: int) -> str:
        # Convert given workers_per_resource to correct CUDA_VISIBLE_DEVICES string.
        if isinstance(workers_per_resource, float):
            # NOTE: We hit this branch when workers_per_resource is set to
            # float, for example 0.5 or 0.25
            if workers_per_resource > 1:
                raise ValueError(
                    "Currently, the default strategy doesn't support workers_per_resource > 1. It is recommended that one should implement a custom strategy in this case."
                )
            # We are round the assigned resource here. This means if workers_per_resource=.4
            # then it will round down to 2. If workers_per_source=0.6, then it will also round up to 2.
            assigned_resource_per_worker = round(1 / workers_per_resource)
            if len(gpus) < assigned_resource_per_worker:
                logger.warning(
                    "Failed to allocate %s GPUs for %s (number of available GPUs < assigned workers per resource [%s])",
                    gpus,
                    worker_index,
                    assigned_resource_per_worker,
                )
                raise IndexError(
                    f"There aren't enough assigned GPU(s) for given worker id '{worker_index}' [required: {assigned_resource_per_worker}]."
                )
            assigned_gpu = gpus[
                assigned_resource_per_worker * worker_index : assigned_resource_per_worker * (worker_index + 1)
            ]
            dev = ",".join(assigned_gpu)
        else:
            idx = worker_index // workers_per_resource
            if idx >= len(gpus):
                raise ValueError(
                    f"Number of available GPU ({gpus}) preceeds the given workers_per_resource {workers_per_resource}"
                )
            dev = str(gpus[idx])
        return dev
