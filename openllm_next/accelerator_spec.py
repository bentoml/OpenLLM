import math
import typing
from types import SimpleNamespace


class ACC_SPEC(SimpleNamespace):
    model: str
    memory_size: float

    def __gt__(self, other):
        return self.memory_size > other.memory_size

    def __eq__(self, other):
        return self.memory_size == other.memory_size


class Resource(SimpleNamespace):
    cpu: int
    memory: float
    gpu: int
    gpu_type: str


ACCELERATOR_SPEC_DICT: dict[str, dict] = {
    "nvidia-gtx-1650": {"model": "GTX 1650", "memory_size": 4.0},
    "nvidia-gtx-1060": {"model": "GTX 1060", "memory_size": 6.0},
    "nvidia-gtx-1080-ti": {"model": "GTX 1080 Ti", "memory_size": 11.0},
    "nvidia-rtx-3060": {"model": "RTX 3060", "memory_size": 12.0},
    "nvidia-rtx-3060-ti": {"model": "RTX 3060 Ti", "memory_size": 8.0},
    "nvidia-rtx-3070-ti": {"model": "RTX 3070 Ti", "memory_size": 8.0},
    "nvidia-rtx-3080": {"model": "RTX 3080", "memory_size": 10.0},
    "nvidia-rtx-3080-ti": {"model": "RTX 3080 Ti", "memory_size": 12.0},
    "nvidia-rtx-3090": {"model": "RTX 3090", "memory_size": 24.0},
    "nvidia-rtx-4070-ti": {"model": "RTX 4070 Ti", "memory_size": 12.0},
    "nvidia-tesla-p4": {"model": "P4", "memory_size": 8.0},
    "nvidia-tesla-p100": {"model": "P100", "memory_size": 16.0},
    "nvidia-tesla-k80": {"model": "K80", "memory_size": 12.0},
    "nvidia-tesla-t4": {"model": "T4", "memory_size": 16.0},
    "nvidia-tesla-v100": {"model": "V100", "memory_size": 16.0},
    "nvidia-l4": {"model": "L4", "memory_size": 24.0},
    "nvidia-tesla-l4": {"model": "L4", "memory_size": 24.0},
    "nvidia-tesla-a10g": {"model": "A10G", "memory_size": 24.0},
    "nvidia-a100-80g": {"model": "A100", "memory_size": 80.0},
    "nvidia-tesla-a100": {"model": "A100", "memory_size": 40.0},
}


ACCELERATOR_SPECS: dict[str, ACC_SPEC] = {
    key: ACC_SPEC(**value) for key, value in ACCELERATOR_SPEC_DICT.items()
}


class MACHINE_SPEC(SimpleNamespace):
    source: str = "local"
    accelerators: list[ACC_SPEC]


def get_local_machine_spec():
    from pynvml import (
        nvmlDeviceGetCount,
        nvmlDeviceGetHandleByIndex,
        nvmlDeviceGetMemoryInfo,
        nvmlDeviceGetName,
        nvmlInit,
        nvmlShutdown,
    )

    nvmlInit()
    device_count = nvmlDeviceGetCount()
    accelerators: list[ACC_SPEC] = []
    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        name = nvmlDeviceGetName(handle)
        memory_info = nvmlDeviceGetMemoryInfo(handle)
        accelerators.append(
            ACC_SPEC(name=name, memory_size=math.ceil(int(memory_info.total) / 1024**3))
        )
    nvmlShutdown()
    return MACHINE_SPEC(accelerators=accelerators, source="local")


def _machine_match_score(resource_spec: Resource, machine: MACHINE_SPEC) -> float:
    if resource_spec.gpu > 0:
        required_gpu = ACCELERATOR_SPECS[resource_spec.gpu_type]
        filtered_accelerators = [
            ac
            for ac in machine.accelerators
            if ac.memory_size >= required_gpu.memory_size
        ]
        if resource_spec.gpu > len(filtered_accelerators):
            return 0.0
        return (
            required_gpu.memory_size
            * resource_spec.gpu
            / sum(ac.memory_size for ac in machine.accelerators)
        )
    if machine.accelerators:
        return 0.01 / sum(ac.memory_size for ac in machine.accelerators)
    return 1.0


def match_machines(
    resource_specs: list[Resource | dict],
    machines: list[MACHINE_SPEC] | None = None,
):
    if resource_specs and isinstance(resource_specs[0], dict):
        resource_specs = [Resource(**spec) for spec in resource_specs]
    if machines is None:
        machines = [get_local_machine_spec()]

    return [
        (resource_spec, machine, _machine_match_score(resource_spec, machine))
        for machine in machines
        for resource_spec in resource_specs
    ]
