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
import logging
import math
import os
import sys
import typing as t

import psutil

import bentoml
import openllm
from bentoml._internal.resource import Resource
from bentoml._internal.resource import get_resource
from bentoml._internal.resource import system_resources
from bentoml._internal.runner.strategy import THREAD_ENVS
from bentoml._internal.runner.strategy import Strategy

from .utils import LazyType
from .utils import ReprMixin


if t.TYPE_CHECKING:
    ListIntStr = list[int | str]
else:
    ListIntStr = list

logger = logging.getLogger(__name__)


class AmdGpuResource(Resource[t.List[int]], resource_id="amd.com/gpu"):
    @classmethod
    def from_spec(cls, spec: int | str | list[str | int]) -> list[int]:
        if not isinstance(spec, (int, str)) and not LazyType(ListIntStr).isinstance(spec):
            raise TypeError("AMD GPU device IDs must be int, str or a list specifing the exact GPUs to use.")
        try:
            if isinstance(spec, int):
                if spec < -1:
                    raise ValueError
                return list(range(spec))
            elif isinstance(spec, str):
                return cls.from_spec(int(spec))
            else:
                return [int(x) for x in spec]
        except ValueError:
            raise openllm.exceptions.OpenLLMException(f"Invalid AMD GPU resource limit '{spec}'. ")

    @classmethod  # type: ignore (overload)
    @functools.lru_cache(maxsize=1)
    def from_system(cls) -> list[int]:
        """Retrieve AMD GPU from system, currently only supports on Linux.
        This assumes that ROCm is setup correctly."""
        if not psutil.LINUX:
            logger.debug("AMD GPU resource is only supported on Linux.")
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

            num = c_uint32(0)
            ret = rocmsmi.rsmi_num_monitor_devices(byref(num))
            if ret == rsmi_status_t.RSMI_STATUS_SUCCESS:
                return list(range(num.value))
            return []
        except Exception as err:
            logger.debug("Failed to setup AMD GPU resource: %s", err)
            return []
        finally:
            sys.path.remove("/opt/rocm/libexec/rocm_smi")

    @classmethod
    def validate(cls, val: list[int]):
        if any(gpu_index < 0 for gpu_index in val):
            raise openllm.exceptions.OpenLLMException(f"Negative GPU device in {val}.")
        if any(gpu_index >= len(cls.from_system()) for gpu_index in val):
            raise openllm.exceptions.OpenLLMException(
                f"GPU device index in {val} is greater than the system available: {cls.from_system()}"
            )


class CascadingResourceStrategy(Strategy, ReprMixin):
    """This is rather an extension of bentoml._internal.runner.strategy.DefaultStrategy
    where we check for NVIDIA GPU resource -> AMD GPU resource -> CPU resource.

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

        # use nvidia gpu
        nvidia_gpus = get_resource(resource_request, "nvidia.com/gpu")
        if nvidia_gpus is not None and len(nvidia_gpus) > 0 and "nvidia.com/gpu" in runnable_class.SUPPORTED_RESOURCES:
            return math.ceil(len(nvidia_gpus) * workers_per_resource)

        # use amd gpu
        amd_gpus = get_resource(resource_request, "amd.com/gpu")
        if amd_gpus is not None and len(amd_gpus) > 0 and "amd.com/gpu" in runnable_class.SUPPORTED_RESOURCES:
            return math.ceil(len(amd_gpus) * workers_per_resource)

        # use CPU
        cpus = get_resource(resource_request, "cpu")
        if cpus is not None and cpus > 0:
            if "cpu" not in runnable_class.SUPPORTED_RESOURCES:
                logger.warning(
                    "No known supported resource available for %s, falling back to using CPU.",
                    runnable_class,
                )

            if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
                if isinstance(workers_per_resource, float):
                    raise ValueError("Fractional CPU multi threading support is not yet supported.")
                return workers_per_resource

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
        """
        Args:
            runnable_class : The runnable class to be run.
            resource_request : The resource request of the runnable.
            worker_index : The index of the worker, start from 0.
        """
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        disabled = cuda_env in ("", "-1")

        environ: dict[str, t.Any] = {}

        if resource_request is None:
            resource_request = system_resources()

        # use nvidia gpu
        nvidia_gpus: list[int] | None = get_resource(resource_request, "nvidia.com/gpu")
        if nvidia_gpus is not None and len(nvidia_gpus) > 0 and "nvidia.com/gpu" in runnable_class.SUPPORTED_RESOURCES:
            dev = cls.transpile_workers_to_cuda_visible_devices(workers_per_resource, nvidia_gpus, worker_index)
            if disabled:
                logger.debug("CUDA_VISIBLE_DEVICES is disabled, %s will not be using GPU.", worker_index)
                environ["CUDA_VISIBLE_DEVICES"] = cuda_env
                return environ
            environ["CUDA_VISIBLE_DEVICES"] = dev
            logger.info(
                "Environ for worker %s: set CUDA_VISIBLE_DEVICES to %s",
                worker_index,
                dev,
            )
            return environ

        # use amd gpu
        amd_gpus = get_resource(resource_request, "amd.com/gpu")
        if amd_gpus is not None and len(amd_gpus) > 0 and "amd.com/gpu" in runnable_class.SUPPORTED_RESOURCES:
            dev = cls.transpile_workers_to_cuda_visible_devices(workers_per_resource, amd_gpus, worker_index)
            if disabled:
                logger.debug("CUDA_VISIBLE_DEVICES is disabled, %s will not be using GPU.", worker_index)
                environ["CUDA_VISIBLE_DEVICES"] = cuda_env
                return environ
            environ["CUDA_VISIBLE_DEVICES"] = dev
            logger.info(
                "Environ for worker %s: set CUDA_VISIBLE_DEVICES to %s",
                worker_index,
                dev,
            )
            return environ

        # use CPU
        cpus = get_resource(resource_request, "cpu")
        if cpus is not None and cpus > 0:
            environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable gpu
            if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
                thread_count = math.ceil(cpus)
                for thread_env in THREAD_ENVS:
                    environ[thread_env] = str(thread_count)
                logger.info(
                    "Environ for worker %d: set CPU thread count to %d",
                    worker_index,
                    thread_count,
                )
                return environ
            else:
                for thread_env in THREAD_ENVS:
                    environ[thread_env] = "1"
                return environ

        return environ

    @staticmethod
    def transpile_workers_to_cuda_visible_devices(
        workers_per_resource: float | int, gpus: list[int], worker_index: int
    ) -> str:
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
            return ",".join(map(str, assigned_gpu))
        else:
            return str(gpus[worker_index // workers_per_resource])
