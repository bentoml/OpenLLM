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

"""Tests utilities for OpenLLM."""

from __future__ import annotations
import contextlib
import logging
import shutil
import subprocess
import typing as t

import bentoml
import openllm


logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    from ._types import LiteralRuntime


@contextlib.contextmanager
def build_bento(
    model: str,
    model_id: str | None = None,
    quantize: t.Literal["int4", "int8", "gptq"] | None = None,
    runtime: t.Literal["ggml", "transformers"] = "transformers",
    cleanup: bool = False,
):
    logger.info("Building BentoML for %s", model)
    bento = openllm.build(model, model_id=model_id, quantize=quantize, runtime=runtime)
    yield bento
    if cleanup:
        logger.info("Deleting %s", bento.tag)
        bentoml.bentos.delete(bento.tag)


@contextlib.contextmanager
def build_container(
    bento: bentoml.Bento | str | bentoml.Tag,
    image_tag: str | None = None,
    cleanup: bool = False,
    **attrs: t.Any,
):
    if isinstance(bento, bentoml.Bento):
        bento_tag = bento.tag
    else:
        bento_tag = bentoml.Tag.from_taglike(bento)

    if image_tag is None:
        image_tag = str(bento_tag)

    executable = shutil.which("docker")
    if not executable:
        raise RuntimeError("docker executable not found")

    try:
        logger.info("Building container for %s", bento_tag)
        bentoml.container.build(
            bento_tag,
            backend="docker",
            image_tag=(image_tag,),
            progress="plain",
            **attrs,
        )
        yield image_tag
    finally:
        if cleanup:
            logger.info("Deleting container %s", image_tag)
            subprocess.check_output([executable, "rmi", "-f", image_tag])


@contextlib.contextmanager
def prepare(
    model: str,
    model_id: str | None = None,
    implementation: LiteralRuntime = "pt",
    deployment_mode: t.Literal["container", "local"] = "local",
    clean_context: contextlib.ExitStack | None = None,
    cleanup: bool = True,
):
    if clean_context is None:
        clean_context = contextlib.ExitStack()
        cleanup = True

    llm = openllm.infer_auto_class(implementation).for_model(model, model_id=model_id, ensure_available=True)
    bento_tag = bentoml.Tag.from_taglike(f"{llm.llm_type}-service:{llm.tag.version}")

    if not bentoml.list(bento_tag):
        bento = clean_context.enter_context(build_bento(model, model_id=model_id, cleanup=cleanup))
    else:
        bento = bentoml.get(bento_tag)

    container_name = f"openllm-{model}-{llm.llm_type}".replace("-", "_")

    if deployment_mode == "container":
        container_name = clean_context.enter_context(build_container(bento, image_tag=container_name, cleanup=cleanup))

    yield container_name
    if cleanup:
        clean_context.close()
