from typing import Dict

import triton
from _typeshed import Incomplete

class CustomizedTritonAutoTuner(triton.KernelInterface):
    configs: Incomplete
    key_idx: Incomplete
    nearest_power_of_two: Incomplete
    cache: Incomplete
    hook: Incomplete
    reset_idx: Incomplete
    arg_names: Incomplete
    early_config_prune: Incomplete
    fn: Incomplete
    def __init__(self, fn, arg_names, configs, key, reset_to_zero, prune_configs_by: Dict = ..., nearest_power_of_two: bool = ...) -> None: ...
    nargs: Incomplete
    bench_time: Incomplete
    configs_timings: Incomplete
    best_config: Incomplete
    def run(self, *args, **kwargs): ...
    def prune_configs(self, kwargs): ...
    def warmup(self, *args, **kwargs) -> None: ...

def autotune(configs, key, prune_configs_by: Incomplete | None = ..., reset_to_zero: Incomplete | None = ..., nearest_power_of_two: bool = ...): ...
