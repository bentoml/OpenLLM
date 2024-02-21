from typing import Any, Type, Dict, Optional, Union, List
import bentoml

def get_resource(resources: Dict[str, Any], resource_kind: str, validate: bool = ...) -> Any: ...

class CascadingResourceStrategy:
  """This is extends the default BentoML strategy where we check for NVIDIA GPU resource -> AMD GPU resource -> CPU resource.

  It also respect CUDA_VISIBLE_DEVICES for both AMD and NVIDIA GPU.
  See https://rocm.docs.amd.com/en/develop/understand/gpu_isolation.html#cuda-visible-devices
  for ROCm's support for CUDA_VISIBLE_DEVICES.

  TODO: Support CloudTPUResource
  """
  @classmethod
  def get_worker_count(
    cls,
    runnable_class: Type[bentoml.Runnable],
    resource_request: Optional[Dict[str, Any]],
    workers_per_resource: float,
  ) -> int:
    """Return the number of workers to be used for the given runnable class.

    Note that for all available GPU, the number of workers will always be 1.
    """
  @classmethod
  def get_worker_env(
    cls,
    runnable_class: type[bentoml.Runnable],
    resource_request: Optional[Dict[str, Any]],
    workers_per_resource: Union[int, float],
    worker_index: int,
  ) -> Dict[str, Any]:
    """Get worker env for this given worker_index.

    Args:
      runnable_class: The runnable class to be run.
      resource_request: The resource request of the runnable.
      workers_per_resource: # of workers per resource.
      worker_index: The index of the worker, start from 0.
    """
  @staticmethod
  def transpile_workers_to_cuda_envvar(
    workers_per_resource: Union[float, int], gpus: List[str], worker_index: int
  ) -> str:
    """Convert given workers_per_resource to correct CUDA_VISIBLE_DEVICES string."""
