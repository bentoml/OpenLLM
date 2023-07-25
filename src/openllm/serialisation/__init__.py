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

import cloudpickle

import openllm
from bentoml._internal.models.model import CUSTOM_OBJECTS_FILENAME

from ..exceptions import OpenLLMException
from ..utils import LazyLoader
from ..utils import LazyModule


if t.TYPE_CHECKING:
    import bentoml
    import transformers

    from .._llm import M
    from .._llm import T
else:
    transformers = LazyLoader("transformers", globals(), "transformers")


def import_model(llm: openllm.LLM[M, T], *decls: t.Any, trust_remote_code: bool, **attrs: t.Any) -> bentoml.Model:
    if llm.runtime == "transformers":
        return openllm.transformers.import_model(llm, *decls, trust_remote_code=trust_remote_code, **attrs)
    elif llm.runtime == "ggml":
        return openllm.ggml.import_model(llm, *decls, trust_remote_code=trust_remote_code, **attrs)
    else:
        raise ValueError(f"Unknown runtime: {llm.config['runtime']}")


def get(llm: openllm.LLM[M, T], auto_import: bool = False) -> bentoml.Model:
    if llm.runtime == "transformers":
        return openllm.transformers.get(llm, auto_import=auto_import)
    elif llm.runtime == "ggml":
        return openllm.ggml.get(llm, auto_import=auto_import)
    else:
        raise ValueError(f"Unknown runtime: {llm.config['runtime']}")


def save_pretrained(llm: openllm.LLM[M, T], save_directory: str, **attrs: t.Any) -> None:
    if llm.runtime == "transformers":
        return openllm.transformers.save_pretrained(llm, save_directory, **attrs)
    elif llm.runtime == "ggml":
        return openllm.ggml.save_pretrained(llm, save_directory, **attrs)
    else:
        raise ValueError(f"Unknown runtime: {llm.config['runtime']}")


def load_model(llm: openllm.LLM[M, T], *decls: t.Any, **attrs: t.Any) -> M:
    if llm.runtime == "transformers":
        return openllm.transformers.load_model(llm, *decls, **attrs)
    elif llm.runtime == "ggml":
        return openllm.ggml.load_model(llm, *decls, **attrs)
    else:
        raise ValueError(f"Unknown runtime: {llm.config['runtime']}")


def load_tokenizer(llm: openllm.LLM[t.Any, T], **tokenizer_attrs: t.Any) -> T:
    """Load the tokenizer from BentoML store.

    By default, it will try to find the bentomodel whether it is in store..
    If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
    """
    from .transformers import infer_tokenizers_class_for_llm

    bentomodel_fs = llm._bentomodel._fs
    if bentomodel_fs.isfile(CUSTOM_OBJECTS_FILENAME):
        with bentomodel_fs.open(CUSTOM_OBJECTS_FILENAME, "rb") as cofile:
            try:
                tokenizer = cloudpickle.load(t.cast("t.IO[bytes]", cofile))["tokenizer"]
            except KeyError:
                # This could happen if users implement their own import_model
                raise OpenLLMException(
                    "Model does not have tokenizer. Make sure to save \
                    the tokenizer within the model via 'custom_objects'.\
                    For example: bentoml.transformers.save_model(..., custom_objects={'tokenizer': tokenizer}))"
                ) from None
    else:
        tokenizer = infer_tokenizers_class_for_llm(llm).from_pretrained(
            bentomodel_fs.getsyspath("/"),
            trust_remote_code=llm.__llm_trust_remote_code__,
            **tokenizer_attrs,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


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
