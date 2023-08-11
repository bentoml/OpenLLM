from __future__ import annotations
import typing as t

import attr
from huggingface_hub import HfApi

if t.TYPE_CHECKING:
  import openllm
  from openllm._llm import M, T

def has_safetensors_weights(model_id: str, revision: str | None = None) -> bool: return any(s.rfilename.endswith(".safetensors") for s in HfApi().model_info(model_id, revision=revision).siblings)

@attr.define(slots=True)
class HfIgnore:
  safetensors = "*.safetensors"
  pt = "*.bin"
  tf = "*.h5"
  flax = "*.msgpack"

  @classmethod
  def ignore_patterns(cls, llm: openllm.LLM[M, T]) -> list[str]:
    if llm.__llm_implementation__ == "vllm": base = [cls.tf, cls.flax, cls.safetensors]
    elif llm.__llm_implementation__ == "tf": base = [cls.flax, cls.pt]
    elif llm.__llm_implementation__ == "flax": base = [cls.tf, cls.pt, cls.safetensors]  # as of current, safetensors is not supported with flax
    else:
      base = [cls.tf, cls.flax]
      if has_safetensors_weights(llm.model_id): base.append(cls.pt)
    # filter out these files, since we probably don't need them for now.
    base.extend(["*.pdf", "*.md", ".gitattributes", "LICENSE.txt"])
    return base
