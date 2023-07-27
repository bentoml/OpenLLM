from __future__ import annotations
import typing as t

import click

import openllm

from .. import termui

if t.TYPE_CHECKING:
    from openllm.bundle.oci import LiteralContainerRegistry

@click.command("build_base_container",
               context_settings=termui.CONTEXT_SETTINGS,
               help="""Base image builder for BentoLLM.

                By default, the base image will include custom kernels (PagedAttention via vllm, FlashAttention-v2, etc.) built with CUDA 11.8, Python 3.9 on Ubuntu22.04.

                Optionally, this can also be pushed directly to remote registry. Currently support ``docker.io``, ``ghcr.io`` and ``quay.io``.
                """)
@click.option("--registry", multiple=True, type=click.Choice(list(openllm.bundle.CONTAINER_NAMES)), help="Target registry to create image tag on.", default=None)
@click.option("--push/--no-push", help="Whether to push to remote repository", is_flag=True, default=False)
def cli(registry: tuple[LiteralContainerRegistry, ...], push: bool): openllm.bundle.build_container(registry, push)
