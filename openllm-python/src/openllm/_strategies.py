# Apache License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import inspect, logging, math, os, sys, types, warnings, typing as t
import psutil, bentoml, openllm_core.utils as coreutils
from bentoml._internal.resource import get_resource, system_resources
from bentoml._internal.runner.strategy import THREAD_ENVS

__all__ = ['CascadingResourceStrategy', 'get_resource']
logger = logging.getLogger(__name__)


def _strtoul(s: str) -> int:
  # Return -1 or positive integer sequence string starts with.
  if not s:
    return -1
  idx = 0
  for idx, c in enumerate(s):
    if not (c.isdigit() or (idx == 0 and c in '+-')):
      break
    if idx + 1 == len(s):
      idx += 1
  # NOTE: idx will be set via enumerate
  return int(s[:idx]) if idx > 0 else -1


def _parse_list_with_prefix(lst: str, prefix: str) -> list[str]:
  rcs = []
  for elem in lst.split(','):
    # Repeated id results in empty set
    if elem in rcs:
      return []
    # Anything other but prefix is ignored
    if not elem.startswith(prefix):
      break
    rcs.append(elem)
  return rcs


def _parse_cuda_visible_devices(default_var: str | None = None, respect_env: bool = True) -> list[str] | None:
  if respect_env:
    spec = os.environ.get('CUDA_VISIBLE_DEVICES', default_var)
    if not spec:
      return None
  else:
    if default_var is None:
      raise ValueError('spec is required to be not None when parsing spec.')
    spec = default_var

  if spec.startswith('GPU-'):
    return _parse_list_with_prefix(spec, 'GPU-')
  if spec.startswith('MIG-'):
    return _parse_list_with_prefix(spec, 'MIG-')
  # XXX: We need to somehow handle cases such as '100m'
  # CUDA_VISIBLE_DEVICES uses something like strtoul
  # which makes `1gpu2,2ampere` is equivalent to `1,2`
  rc: list[int] = []
  for el in spec.split(','):
    x = _strtoul(el.strip())
    # Repeated ordinal results in empty set
    if x in rc:
      return []
    # Negative value aborts the sequence
    if x < 0:
      break
    rc.append(x)
  return [str(i) for i in rc]


def _raw_device_uuid_nvml() -> list[str] | None:
  from ctypes import CDLL, byref, c_int, c_void_p, create_string_buffer

  try:
    nvml_h = CDLL('libnvidia-ml.so.1')
  except Exception:
    warnings.warn('Failed to find nvidia binding', stacklevel=3)
    return None

  rc = nvml_h.nvmlInit()
  if rc != 0:
    warnings.warn("Can't initialize NVML", stacklevel=3)
    return None
  dev_count = c_int(-1)
  rc = nvml_h.nvmlDeviceGetCount_v2(byref(dev_count))
  if rc != 0:
    warnings.warn('Failed to get available device from system.', stacklevel=3)
    return None
  uuids = []
  for idx in range(dev_count.value):
    dev_id = c_void_p()
    rc = nvml_h.nvmlDeviceGetHandleByIndex_v2(idx, byref(dev_id))
    if rc != 0:
      warnings.warn(f'Failed to get device handle for {idx}', stacklevel=3)
      return None
    buf_len = 96
    buf = create_string_buffer(buf_len)
    rc = nvml_h.nvmlDeviceGetUUID(dev_id, buf, buf_len)
    if rc != 0:
      warnings.warn(f'Failed to get device UUID for {idx}', stacklevel=3)
      return None
    uuids.append(buf.raw.decode('ascii').strip('\0'))
  del nvml_h
  return uuids


class _ResourceMixin:
  @staticmethod
  def from_system(cls) -> list[str]:
    visible_devices = _parse_cuda_visible_devices()
    if visible_devices is None:
      if cls.resource_id == 'amd.com/gpu':
        if not psutil.LINUX:
          return []
        # ROCm does not currently have the rocm_smi wheel.
        # So we need to use the ctypes bindings directly.
        # we don't want to use CLI because parsing is a pain.
        # TODO: Use tinygrad/gpuctypes
        rocm_path = os.environ.get('ROCM_PATH', '/opt/rocm')
        sys.path.append(rocm_path + '/libexec/rocm_smi')
        try:
          from ctypes import byref, c_uint32

          # refers to https://github.com/RadeonOpenCompute/rocm_smi_lib/blob/master/python_smi_tools/rsmiBindings.py
          from rsmiBindings import rocmsmi, rsmi_status_t

          ret = rocmsmi.rsmi_init(0)
          device_count = c_uint32(0)
          ret = rocmsmi.rsmi_num_monitor_devices(byref(device_count))
          if ret == rsmi_status_t.RSMI_STATUS_SUCCESS:
            return [str(i) for i in range(device_count.value)]
          return []
        # In this case the binary is not found, returning empty list
        except (ModuleNotFoundError, ImportError):
          return []
        finally:
          sys.path.remove(rocm_path + '/libexec/rocm_smi')
      else:
        try:
          from cuda import cuda

          cuda.cuInit(0)
          _, dev = cuda.cuDeviceGetCount()
          return [str(i) for i in range(dev)]
        except (ImportError, RuntimeError, AttributeError):
          return []
    return visible_devices

  @staticmethod
  def from_spec(cls, spec) -> list[str]:
    if isinstance(spec, int):
      if spec in (-1, 0):
        return []
      if spec < -1:
        raise ValueError('Spec cannot be < -1.')
      return [str(i) for i in range(spec)]
    elif isinstance(spec, str):
      if not spec:
        return []
      if spec.isdigit():
        spec = ','.join([str(i) for i in range(_strtoul(spec))])
      return _parse_cuda_visible_devices(spec, respect_env=False)
    elif isinstance(spec, list):
      return [str(x) for x in spec]
    else:
      raise TypeError(
        f"'{cls.__name__}.from_spec' only supports parsing spec of type int, str, or list, got '{type(spec)}' instead."
      )

  @staticmethod
  def validate(cls, val: list[t.Any]) -> None:
    if cls.resource_id == 'amd.com/gpu':
      raise RuntimeError(
        "AMD GPU validation is not yet supported. Make sure to call 'get_resource(..., validate=False)'"
      )
    if not all(isinstance(i, str) for i in val):
      raise ValueError('Input list should be all string type.')

    try:
      from cuda import cuda

      err, *_ = cuda.cuInit(0)
      if err != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError('Failed to initialise CUDA runtime binding.')
      # correctly parse handle
      for el in val:
        if el.startswith(('GPU-', 'MIG-')):
          uuids = _raw_device_uuid_nvml()
          if uuids is None:
            raise ValueError('Failed to parse available GPUs UUID')
          if el not in uuids:
            raise ValueError(f'Given UUID {el} is not found with available UUID (available: {uuids})')
        elif el.isdigit():
          err, _ = cuda.cuDeviceGet(int(el))
          if err != cuda.CUresult.CUDA_SUCCESS:
            raise ValueError(f'Failed to get device {el}')
    except (ImportError, RuntimeError):
      pass


def _make_resource_class(name: str, resource_kind: str, docstring: str) -> type[bentoml.Resource[t.List[str]]]:
  return types.new_class(
    name,
    (bentoml.Resource[t.List[str]], coreutils.ReprMixin),
    {'resource_id': resource_kind},
    lambda ns: ns.update({
      'resource_id': resource_kind,
      'from_spec': classmethod(_ResourceMixin.from_spec),
      'from_system': classmethod(_ResourceMixin.from_system),  #
      'validate': classmethod(_ResourceMixin.validate),
      '__repr_keys__': property(lambda _: {'resource_id'}),  #
      '__doc__': inspect.cleandoc(docstring),
      '__module__': 'openllm._strategies',  #
    }),
  )


NvidiaGpuResource = _make_resource_class(
  'NvidiaGpuResource',
  'nvidia.com/gpu',
  """NVIDIA GPU resource.
    This is a modified version of internal's BentoML's NvidiaGpuResource
    where it respects and parse CUDA_VISIBLE_DEVICES correctly.""",
)
AmdGpuResource = _make_resource_class(
  'AmdGpuResource',
  'amd.com/gpu',
  """AMD GPU resource.
    Since ROCm will respect CUDA_VISIBLE_DEVICES, the behaviour of from_spec, from_system are similar to
    ``NvidiaGpuResource``. Currently ``validate`` is not yet supported.""",
)


class CascadingResourceStrategy(bentoml.Strategy, coreutils.ReprMixin):
  @classmethod
  def get_worker_count(cls, runnable_class, resource_request, workers_per_resource):
    if resource_request is None:
      resource_request = system_resources()
    # use NVIDIA
    kind = 'nvidia.com/gpu'
    nvidia_req = get_resource(resource_request, kind)
    if nvidia_req is not None:
      return 1
    # use AMD
    kind = 'amd.com/gpu'
    amd_req = get_resource(resource_request, kind, validate=False)
    if amd_req is not None:
      return 1
    # use CPU
    cpus = get_resource(resource_request, 'cpu')
    if cpus is not None and cpus > 0:
      if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
        if isinstance(workers_per_resource, float) and workers_per_resource < 1.0:
          raise ValueError('Fractional CPU multi threading support is not yet supported.')
        return int(workers_per_resource)
      return math.ceil(cpus) * workers_per_resource
    # this should not be reached by user since we always read system resource as default
    raise ValueError(
      f'No known supported resource available for {runnable_class}. Please check your resource request. Leaving it blank will allow BentoML to use system resources.'
    )

  @classmethod
  def get_worker_env(cls, runnable_class, resource_request, workers_per_resource, worker_index):
    cuda_env = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    disabled = cuda_env in ('', '-1')
    environ = {}

    if resource_request is None:
      resource_request = system_resources()
    # use NVIDIA
    kind = 'nvidia.com/gpu'
    typ = get_resource(resource_request, kind)
    if typ is not None and len(typ) > 0 and kind in runnable_class.SUPPORTED_RESOURCES:
      if disabled:
        environ['CUDA_VISIBLE_DEVICES'] = cuda_env
        return environ
      environ['CUDA_VISIBLE_DEVICES'] = cls.transpile_workers_to_cuda_envvar(workers_per_resource, typ, worker_index)
      return environ
    # use AMD
    kind = 'amd.com/gpu'
    typ = get_resource(resource_request, kind, validate=False)
    if typ is not None and len(typ) > 0 and kind in runnable_class.SUPPORTED_RESOURCES:
      if disabled:
        environ['CUDA_VISIBLE_DEVICES'] = cuda_env
        return environ
      environ['CUDA_VISIBLE_DEVICES'] = cls.transpile_workers_to_cuda_envvar(workers_per_resource, typ, worker_index)
      return environ
    # use CPU
    cpus = get_resource(resource_request, 'cpu')
    if cpus is not None and cpus > 0:
      environ['CUDA_VISIBLE_DEVICES'] = '-1'  # disable gpu
      if runnable_class.SUPPORTS_CPU_MULTI_THREADING:
        thread_count = math.ceil(cpus)
        for thread_env in THREAD_ENVS:
          environ[thread_env] = os.environ.get(thread_env, str(thread_count))
        return environ
      for thread_env in THREAD_ENVS:
        environ[thread_env] = os.environ.get(thread_env, '1')
      return environ
    return environ

  @staticmethod
  def transpile_workers_to_cuda_envvar(workers_per_resource, gpus, worker_index):
    # Convert given workers_per_resource to correct CUDA_VISIBLE_DEVICES string.
    if isinstance(workers_per_resource, float):
      # NOTE: We hit this branch when workers_per_resource is set to float, for example 0.5 or 0.25
      if workers_per_resource > 1:
        raise ValueError('workers_per_resource > 1 is not supported.')
      # We are round the assigned resource here. This means if workers_per_resource=.4
      # then it will round down to 2. If workers_per_source=0.6, then it will also round up to 2.
      assigned_resource_per_worker = round(1 / workers_per_resource)
      if len(gpus) < assigned_resource_per_worker:
        logger.warning(
          'Failed to allocate %s GPUs for %s (number of available GPUs < assigned workers per resource [%s])',
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
      dev = ','.join(assigned_gpu)
    else:
      idx = worker_index // workers_per_resource
      if idx >= len(gpus):
        raise ValueError(
          f'Number of available GPU ({gpus}) preceeds the given workers_per_resource {workers_per_resource}'
        )
      dev = str(gpus[idx])
    return dev
