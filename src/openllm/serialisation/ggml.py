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
"""Serialisation related implementation for GGML-based implementation.

This requires ctransformers to be installed.
"""
from __future__ import annotations
import typing as t

import cloudpickle

import bentoml
from bentoml._internal.models.model import CUSTOM_OBJECTS_FILENAME

from ..exceptions import OpenLLMException
from ..utils import LazyLoader


if t.TYPE_CHECKING:
    import openllm
    import transformers

    from .._llm import M
    from .._llm import T
    from .._types import ModelProtocol
    from .._types import TokenizerProtocol
else:
    transformers = LazyLoader("transformers", globals(), "transformers")


def import_model(
    llm: openllm.LLM[t.Any, t.Any],
    *decls: t.Any,
    trust_remote_code: bool = True,
    **attrs: t.Any,
) -> bentoml.Model:
    raise NotImplementedError("Currently work in progress.")


def get(llm: openllm.LLM[t.Any, t.Any], auto_import: bool = False) -> bentoml.Model:
    """Return an instance of ``bentoml.Model`` from given LLM instance.

    By default, it will try to check the model in the local store.
    If model is not found, and ``auto_import`` is set to True, it will try to import the model from HuggingFace Hub.

    Otherwise, it will raises a ``bentoml.exceptions.NotFound``.
    """
    try:
        model = bentoml.models.get(llm.tag)
        if model.info.module not in ("openllm.serialisation.ggml", __name__):
            raise bentoml.exceptions.NotFound(
                f"Model {model.tag} was saved with module {model.info.module}, not loading with 'openllm.serialisation.transformers'."
            )
        if "runtime" in model.info.labels and model.info.labels["runtime"] != llm.runtime:
            raise OpenLLMException(
                f"Model {model.tag} was saved with runtime {model.info.labels['runtime']}, not loading with {llm.runtime}."
            )
        return model
    except bentoml.exceptions.NotFound:
        if auto_import:
            return import_model(llm, trust_remote_code=llm.__llm_trust_remote_code__)
        raise


def load_model(llm: openllm.LLM[M, t.Any], *decls: t.Any, **attrs: t.Any) -> ModelProtocol[M]:
    """Load the model from BentoML store.

    By default, it will try to find check the model in the local store.
    If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
    """
    raise NotImplementedError("Currently work in progress.")


def load_tokenizer(llm: openllm.LLM[t.Any, T], **tokenizer_attrs: t.Any) -> TokenizerProtocol[T]:
    """Load the tokenizer from BentoML store.

    By default, it will try to find the bentomodel whether it is in store..
    If model is not found, it will raises a ``bentoml.exceptions.NotFound``.
    """
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
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            bentomodel_fs.getsyspath("/"),
            trust_remote_code=llm.__llm_trust_remote_code__,
            **tokenizer_attrs,
        )
    return t.cast("TokenizerProtocol[T]", tokenizer)


def save_pretrained(llm: openllm.LLM[t.Any, t.Any], save_directory: str, **attrs: t.Any) -> None:
    raise NotImplementedError("Currently work in progress.")
