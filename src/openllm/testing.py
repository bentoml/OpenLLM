"""Tests utilities for OpenLLM.
"""

from __future__ import annotations

import typing as t
import bentoml
import subprocess
import openllm
import contextlib
import logging

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def build_bento(
    model: str,
    model_id: str | None = None,
    quantize: t.Literal["int4", "int8", "gptq"] | None = None,
    runtime: t.Literal["ggml", "transformers"] = "transformers",
    cleanup: bool = True,
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
    cleanup: bool = True,
    backend: t.LiteralString = "docker",
    **attrs: t.Any,
):
    if isinstance(bento, bentoml.Bento):
        bento_tag = bento.tag
    else:
        bento_tag = bentoml.Tag.from_taglike(bento)

    if image_tag is None:
        image_tag = str(bento_tag)

    try:
        logger.info("Building container for %s", bento_tag)
        bentoml.container.build(bento_tag, backend=backend, image_tag=(image_tag,), progress="plain", **attrs)
        yield image_tag
    finally:
        if cleanup:
            logger.info("Deleting container %s", image_tag)
            subprocess.call([backend, "rmi", image_tag])
