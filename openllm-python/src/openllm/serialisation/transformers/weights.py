from __future__ import annotations
import attr, traceback, functools, pathlib, typing as t
from huggingface_hub import HfApi
from openllm_core.exceptions import Error
from openllm_core.utils import resolve_filepath, validate_is_path

if t.TYPE_CHECKING:
  from huggingface_hub.hf_api import ModelInfo as HfModelInfo

__global_inst__ = None
__cached_id__: dict[str, HfModelInfo] = dict()


def Client() -> HfApi:
  global __global_inst__
  if __global_inst__ is None:
    __global_inst__ = HfApi()
  return __global_inst__


def ModelInfo(model_id: str, revision: str | None = None) -> HfModelInfo:
  if model_id in __cached_id__:
    return __cached_id__[model_id]
  try:
    __cached_id__[model_id] = Client().model_info(model_id, revision=revision)
    return __cached_id__[model_id]
  except Exception as err:
    traceback.print_exc()
    raise Error(f'Failed to fetch {model_id} from huggingface.co') from err


def has_weights(model_id: str, revision: str | None = None, *, extensions: str) -> bool:
  if validate_is_path(model_id):
    return next((True for _ in pathlib.Path(resolve_filepath(model_id)).glob(f'*.{extensions}')), False)
  return any(s.rfilename.endswith(f'.{extensions}') for s in ModelInfo(model_id, revision=revision).siblings)


has_safetensors_weights = functools.partial(has_weights, extensions='safetensors')
has_pt_weights = functools.partial(has_weights, extensions='pt')


@attr.define(slots=True)
class HfIgnore:
  safetensors = '*.safetensors'
  pt = '*.bin'
  tf = '*.h5'
  flax = '*.msgpack'
  gguf = '*.gguf'

  @classmethod
  def ignore_patterns(cls, backend, model_id) -> list[str]:
    if backend in {'vllm', 'pt'}:
      base = [cls.tf, cls.flax, cls.gguf]
      if has_safetensors_weights(model_id):
        base.extend([cls.pt, '*.pt'])
      elif has_pt_weights(model_id):
        base.extend([cls.safetensors, cls.pt])
      else:
        base.append(cls.safetensors)
    elif backend == 'ggml':
      base = [cls.tf, cls.flax, cls.pt, cls.safetensors]
    else:
      raise ValueError('Unknown backend (should never happen at all.)')
    # filter out these files, since we probably don't need them for now.
    base.extend(['*.pdf', '*.md', '.gitattributes', 'LICENSE.txt', 'Notice'])
    return base
