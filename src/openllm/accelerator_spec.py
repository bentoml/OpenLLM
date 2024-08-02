from __future__ import annotations

import functools
import math
import typing
from types import SimpleNamespace

import psutil

from openllm.common import BentoInfo, DeploymentTarget, output


class Accelerator(SimpleNamespace):
    model: str
    memory_size: float

    def __gt__(self, other):
        return self.memory_size > other.memory_size

    def __eq__(self, other):
        return self.memory_size == other.memory_size

    def __repr__(self):
        return f'{self.model}({self.memory_size}GB)'


class Resource(SimpleNamespace):
    cpu: int = 0
    memory: float
    gpu: int = 0
    gpu_type: str = ''

    def __hash__(self):
        return hash((self.cpu, self.memory, self.gpu, self.gpu_type))

    def __bool__(self):
        return any(value is not None for value in self.__dict__.values())


ACCELERATOR_SPEC_DICT: dict[str, dict] = {
    'nvidia-gtx-1650': {'model': 'GTX 1650', 'memory_size': 4.0},
    'nvidia-gtx-1060': {'model': 'GTX 1060', 'memory_size': 6.0},
    'nvidia-gtx-1080-ti': {'model': 'GTX 1080 Ti', 'memory_size': 11.0},
    'nvidia-rtx-3060': {'model': 'RTX 3060', 'memory_size': 12.0},
    'nvidia-rtx-3060-ti': {'model': 'RTX 3060 Ti', 'memory_size': 8.0},
    'nvidia-rtx-3070-ti': {'model': 'RTX 3070 Ti', 'memory_size': 8.0},
    'nvidia-rtx-3080': {'model': 'RTX 3080', 'memory_size': 10.0},
    'nvidia-rtx-3080-ti': {'model': 'RTX 3080 Ti', 'memory_size': 12.0},
    'nvidia-rtx-3090': {'model': 'RTX 3090', 'memory_size': 24.0},
    'nvidia-rtx-4070-ti': {'model': 'RTX 4070 Ti', 'memory_size': 12.0},
    'nvidia-tesla-p4': {'model': 'P4', 'memory_size': 8.0},
    'nvidia-tesla-p100': {'model': 'P100', 'memory_size': 16.0},
    'nvidia-tesla-k80': {'model': 'K80', 'memory_size': 12.0},
    'nvidia-tesla-t4': {'model': 'T4', 'memory_size': 16.0},
    'nvidia-tesla-v100': {'model': 'V100', 'memory_size': 16.0},
    'nvidia-l4': {'model': 'L4', 'memory_size': 24.0},
    'nvidia-tesla-l4': {'model': 'L4', 'memory_size': 24.0},
    'nvidia-tesla-a10g': {'model': 'A10G', 'memory_size': 24.0},
    'nvidia-a100-80g': {'model': 'A100', 'memory_size': 80.0},
    'nvidia-a100-80gb': {'model': 'A100', 'memory_size': 80.0},
    'nvidia-tesla-a100': {'model': 'A100', 'memory_size': 40.0},
}

ACCELERATOR_SPECS: dict[str, Accelerator] = {key: Accelerator(**value) for key, value in ACCELERATOR_SPEC_DICT.items()}


@functools.lru_cache
def get_local_machine_spec():
    if psutil.MACOS:
        return DeploymentTarget(accelerators=[], source='local', platform='macos')

    if psutil.WINDOWS:
        platform = 'windows'
    elif psutil.LINUX:
        platform = 'linux'
    else:
        raise NotImplementedError('Unsupported platform')

    from pynvml import (
        nvmlDeviceGetCount,
        nvmlDeviceGetCudaComputeCapability,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlInit,
        nvmlShutdown,
    )

    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        accelerators: list[Accelerator] = []
        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            name = nvmlDeviceGetName(handle)
            memory_info = nvmlDeviceGetMemoryInfo(handle)
            accelerators.append(Accelerator(model=name, memory_size=math.ceil(int(memory_info.total) / 1024**3)))
            compute_capability = nvmlDeviceGetCudaComputeCapability(handle)
            if compute_capability < (7, 5):
                output(
                    f'GPU {name} with compute capability {compute_capability} '
                    'may not be supported, 7.5 or higher is recommended. check '
                    'https://developer.nvidia.com/cuda-gpus for more information',
                    style='yellow',
                )
        nvmlShutdown()
        return DeploymentTarget(accelerators=accelerators, source='local', platform=platform)
    except Exception as e:
        output(
            'Failed to get local GPU info. Ensure nvidia driver is installed to enable local GPU deployment',
            style='yellow',
        )
        output(f'Error: {e}', style='red', level=20)
        return DeploymentTarget(accelerators=[], source='local', platform=platform)


@functools.lru_cache()
def can_run(bento: typing.Union[Resource, BentoInfo], target: typing.Optional[DeploymentTarget] = None) -> float:
    """
    Calculate if the bento can be deployed on the target.
    """
    if target is None:
        target = get_local_machine_spec()

    resource_spec = Resource(**(bento.bento_yaml['services'][0]['config'].get('resources', {})))
    labels = bento.bento_yaml.get('labels', {})
    platforms = labels.get('platforms', 'linux').split(',')

    if target.platform not in platforms:
        return 0.0

    # return 1.0 if no resource is specified
    if not resource_spec:
        return 0.5

    if resource_spec.gpu > 0:
        required_gpu = ACCELERATOR_SPECS[resource_spec.gpu_type]
        filtered_accelerators = [ac for ac in target.accelerators if ac.memory_size >= required_gpu.memory_size]
        if resource_spec.gpu > len(filtered_accelerators):
            return 0.0
        return required_gpu.memory_size * resource_spec.gpu / sum(ac.memory_size for ac in target.accelerators)
    if target.accelerators:
        return 0.01 / sum(ac.memory_size for ac in target.accelerators)
    return 1.0
