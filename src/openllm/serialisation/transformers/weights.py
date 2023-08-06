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
import typing as t

import attr
from huggingface_hub import HfApi

if t.TYPE_CHECKING:
  import openllm

  from ..._llm import M
  from ..._llm import T

def has_safetensors_weights(model_id: str, revision: str | None = None) -> bool:
  return any(s.rfilename.endswith(".safetensors") for s in HfApi().model_info(model_id, revision=revision).siblings)

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
