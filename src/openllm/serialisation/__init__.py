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
"""Serialisation utilities for OpenLLM.

Currently supports transformers for PyTorch, Tensorflow and Flax.

Currently, GGML format is working in progress.

## Usage

```python
import openllm

llm = openllm.AutoLLM.for_model("dolly-v2")
llm.save_pretrained("./path/to/local-dolly")
```

To use different runtime, specify directly in the `for_model` method:

```python
import openllm

llm = openllm.AutoLLM.for_model("dolly-v2", runtime='ggml')
llm.save_pretrained("./path/to/local-dolly")
```
"""

from __future__ import annotations
import typing as t

import openllm

from .constants import HUB_ATTRS
from ..utils import LazyModule


if t.TYPE_CHECKING:
    import bentoml

    from .._llm import M
    from .._llm import T
    from .._types import ModelProtocol
    from .._types import TokenizerProtocol


def import_model(
    llm: openllm.LLM[t.Any, t.Any],
    *decls: t.Any,
    trust_remote_code: bool,
    **attrs: t.Any,
) -> bentoml.Model:
    if llm.runtime == "transformers":
        return openllm.transformers.import_model(llm, *decls, trust_remote_code=trust_remote_code, **attrs)
    elif llm.runtime == "ggml":
        return openllm.ggml.import_model(llm, *decls, trust_remote_code=trust_remote_code, **attrs)
    else:
        raise ValueError(f"Unknown runtime: {llm.config['runtime']}")


def get(llm: openllm.LLM[t.Any, t.Any], auto_import: bool = False) -> bentoml.Model:
    if llm.runtime == "transformers":
        return openllm.transformers.get(llm, auto_import=auto_import)
    elif llm.runtime == "ggml":
        return openllm.ggml.get(llm, auto_import=auto_import)
    else:
        raise ValueError(f"Unknown runtime: {llm.config['runtime']}")


def save_pretrained(llm: openllm.LLM[t.Any, t.Any], save_directory: str, **attrs: t.Any) -> None:
    if llm.runtime == "transformers":
        return openllm.transformers.save_pretrained(llm, save_directory, **attrs)
    elif llm.runtime == "ggml":
        return openllm.ggml.save_pretrained(llm, save_directory, **attrs)
    else:
        raise ValueError(f"Unknown runtime: {llm.config['runtime']}")


def load_model(llm: openllm.LLM[M, t.Any], *decls: t.Any, **attrs: t.Any) -> ModelProtocol[M]:
    if llm.__llm_custom_load__:
        hub_attrs = {k: attrs.pop(k) for k in HUB_ATTRS if k in attrs}
        return llm.load_model(llm.tag, *decls, **hub_attrs, **attrs)
    if llm.runtime == "transformers":
        return openllm.transformers.load_model(llm, *decls, **attrs)
    elif llm.runtime == "ggml":
        return openllm.ggml.load_model(llm, *decls, **attrs)
    else:
        raise ValueError(f"Unknown runtime: {llm.config['runtime']}")


def load_tokenizer(llm: openllm.LLM[t.Any, T]) -> TokenizerProtocol[T]:
    if llm.__llm_custom_tokenizer__:
        (_, _), tokenizer_attrs = llm.llm_parameters
        return llm.load_tokenizer(llm.tag, **tokenizer_attrs)
    elif llm.runtime == "transformers":
        return openllm.transformers.load_tokenizer(llm)
    elif llm.runtime == "ggml":
        return openllm.ggml.load_tokenizer(llm)
    else:
        raise ValueError(f"Unknown runtime: {llm.config['runtime']}")


_extras = {
    "get": get,
    "import_model": import_model,
    "save_pretrained": save_pretrained,
    "load_model": load_model,
    "load_tokenizer": load_tokenizer,
}

_import_structure: dict[str, list[str]] = {"ggml": [], "transformers": []}

if t.TYPE_CHECKING:
    from . import ggml as ggml
    from . import transformers as transformers
else:
    import sys

    sys.modules[__name__] = LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects=_extras,
    )
