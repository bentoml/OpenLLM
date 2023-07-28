from __future__ import annotations
import typing as t

import click
import orjson

import openllm

from .. import termui
from .._factory import machine_option

if t.TYPE_CHECKING:
    from openllm.bundle.oci import LiteralContainerRegistry

@click.command("build_base_container",
               context_settings=termui.CONTEXT_SETTINGS,
               help="""Base image builder for BentoLLM.

                By default, the base image will include custom kernels (PagedAttention via vllm, FlashAttention-v2, etc.) built with CUDA 11.8, Python 3.9 on Ubuntu22.04.

                Optionally, this can also be pushed directly to remote registry. Currently support ``docker.io``, ``ghcr.io`` and ``quay.io``.

                If '--machine' is passed, then it will run the process quietly, and output a JSON to the current running terminal.
                """)
@click.option("--registry", multiple=True, type=click.Choice(list(openllm.bundle.CONTAINER_NAMES)), help="Target registry to create image tag on.", default=None)
@click.option("--push/--no-push", help="Whether to push to remote repository", is_flag=True, default=False)
@machine_option
def cli(registry: tuple[LiteralContainerRegistry, ...] | None, push: bool, machine: bool) -> dict[str, str]:
    mapping = openllm.bundle.build_container(registry, push, machine)
    if machine: termui.echo(orjson.dumps(mapping, option=orjson.OPT_INDENT_2).decode(), fg="white")
    return mapping
