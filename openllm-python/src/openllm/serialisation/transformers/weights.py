from __future__ import annotations
import typing as t

import attr

from huggingface_hub import HfApi

if t.TYPE_CHECKING:
  import openllm

  from openllm_core._typing_compat import M
  from openllm_core._typing_compat import T

def has_safetensors_weights(model_id: str, revision: str | None = None) -> bool:
  return any(s.rfilename.endswith('.safetensors') for s in HfApi().model_info(model_id, revision=revision).siblings)

@attr.define(slots=True)
class HfIgnore:
  safetensors = '*.safetensors'
  pt = '*.bin'
  tf = '*.h5'
  flax = '*.msgpack'
  gguf = '*.gguf'

  @classmethod
  def ignore_patterns(cls, llm: openllm.LLM[M, T]) -> list[str]:
    if llm.__llm_backend__ == 'vllm':
      base = [cls.tf, cls.flax, cls.gguf]
      if has_safetensors_weights(llm.model_id) or llm._serialisation == 'safetensors': base.append(cls.pt)
      else: base.append(cls.safetensors)
    elif llm.__llm_backend__ == 'tf': base = [cls.flax, cls.pt, cls.gguf]
    elif llm.__llm_backend__ == 'flax':
      base = [cls.tf, cls.pt, cls.safetensors, cls.gguf]  # as of current, safetensors is not supported with flax
    elif llm.__llm_backend__ == 'pt':
      base = [cls.tf, cls.flax, cls.gguf]
      if has_safetensors_weights(llm.model_id) or llm._serialisation == 'safetensors': base.append(cls.pt)
      else: base.append(cls.safetensors)
    elif llm.__llm_backend__ == 'ggml':
      base = [cls.tf, cls.flax, cls.pt, cls.safetensors]
    else:
      raise ValueError('Unknown backend (should never happen at all.)')
    # filter out these files, since we probably don't need them for now.
    base.extend(['*.pdf', '*.md', '.gitattributes', 'LICENSE.txt', 'Notice'])
    return base
