from __future__ import annotations

import functools, math, re, typing
import psutil, pydantic

from pydantic import BeforeValidator
from typing_extensions import override
from openllm.common import BentoInfo, DeploymentTarget, output, Accelerator


def parse_memory_string(v: typing.Any) -> typing.Any:
    """Parse memory strings like "60Gi" into float."""
    if isinstance(v, str):
        match = re.match(r"(\d+(\.\d+)?)\s*Gi$", v, re.IGNORECASE)
        if match:
            return float(match.group(1))
    # Pass other types (including numbers or other strings for standard float conversion) through
    return v


class Resource(pydantic.BaseModel):
    memory: typing.Annotated[float, BeforeValidator(parse_memory_string)] = 0.0
    cpu: int = 0
    gpu: int = 0
    gpu_type: str = ''

    @override
    def __hash__(self) -> int:
        return hash((self.cpu, self.memory, self.gpu, self.gpu_type))

    def __bool__(self) -> bool:
        return any(value is not None for value in self.__dict__.values())


ACCELERATOR_SPECS: dict[str, Accelerator] = {
    'nvidia-gtx-1650': Accelerator(model='GTX 1650', memory_size=4.0),
    'nvidia-gtx-1060': Accelerator(model='GTX 1060', memory_size=6.0),
    'nvidia-gtx-1080-ti': Accelerator(model='GTX 1080 Ti', memory_size=11.0),
    'nvidia-rtx-3060': Accelerator(model='RTX 3060', memory_size=12.0),
    'nvidia-rtx-3060-ti': Accelerator(model='RTX 3060 Ti', memory_size=8.0),
    'nvidia-rtx-3070-ti': Accelerator(model='RTX 3070 Ti', memory_size=8.0),
    'nvidia-rtx-3080': Accelerator(model='RTX 3080', memory_size=10.0),
    'nvidia-rtx-3080-ti': Accelerator(model='RTX 3080 Ti', memory_size=12.0),
    'nvidia-rtx-3090': Accelerator(model='RTX 3090', memory_size=24.0),
    'nvidia-rtx-4070-ti': Accelerator(model='RTX 4070 Ti', memory_size=12.0),
    'nvidia-tesla-p4': Accelerator(model='P4', memory_size=8.0),
    'nvidia-tesla-p100': Accelerator(model='P100', memory_size=16.0),
    'nvidia-tesla-k80': Accelerator(model='K80', memory_size=12.0),
    'nvidia-tesla-t4': Accelerator(model='T4', memory_size=16.0),
    'nvidia-tesla-v100': Accelerator(model='V100', memory_size=16.0),
    'nvidia-l4': Accelerator(model='L4', memory_size=24.0),
    'nvidia-tesla-l4': Accelerator(model='L4', memory_size=24.0),
    'nvidia-tesla-a10g': Accelerator(model='A10G', memory_size=24.0),
    'nvidia-a100-80g': Accelerator(model='A100', memory_size=80.0),
    'nvidia-a100-80gb': Accelerator(model='A100', memory_size=80.0),
    'nvidia-tesla-a100': Accelerator(model='A100', memory_size=40.0),
}


@functools.lru_cache
def get_local_machine_spec() -> DeploymentTarget:
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


@functools.lru_cache(typed=True)
def can_run(bento: BentoInfo, target: DeploymentTarget | None = None) -> float:
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
