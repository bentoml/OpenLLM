from __future__ import annotations
import typing as t

import click
import orjson

import openllm

from openllm.cli import termui
from openllm.cli._factory import container_registry_option
from openllm.cli._factory import machine_option

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import LiteralContainerRegistry
  from openllm_core._typing_compat import LiteralContainerVersionStrategy

@click.command('build_base_container',
               context_settings=termui.CONTEXT_SETTINGS,
               help='''Base image builder for BentoLLM.

                By default, the base image will include custom kernels (PagedAttention via vllm, FlashAttention-v2, etc.) built with CUDA 11.8, Python 3.9 on Ubuntu22.04.
                Optionally, this can also be pushed directly to remote registry. Currently support ``docker.io``, ``ghcr.io`` and ``quay.io``.

                \b
                If '--machine' is passed, then it will run the process quietly, and output a JSON to the current running terminal.
                This command is only useful for debugging and for building custom base image for extending BentoML with custom base images and custom kernels.

                Note that we already release images on our CI to ECR and GHCR, so you don't need to build it yourself.
                ''')
@container_registry_option
@click.option('--version-strategy', type=click.Choice(['release', 'latest', 'nightly']), default='nightly', help='Version strategy to use for tagging the image.')
@click.option('--push/--no-push', help='Whether to push to remote repository', is_flag=True, default=False)
@machine_option
def cli(container_registry: tuple[LiteralContainerRegistry, ...] | None, version_strategy: LiteralContainerVersionStrategy, push: bool, machine: bool) -> dict[str, str]:
  mapping = openllm.bundle.build_container(container_registry, version_strategy, push, machine)
  if machine: termui.echo(orjson.dumps(mapping, option=orjson.OPT_INDENT_2).decode(), fg='white')
  return mapping
