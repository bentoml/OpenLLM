from __future__ import annotations
import pathlib
import shutil
import subprocess
import typing as t

import click
import orjson

import bentoml
import openllm
from openllm_cli import termui
from openllm_cli._factory import container_registry_option, machine_option
from openllm_core.utils import get_debug_mode, pkg

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import LiteralContainerRegistry, LiteralContainerVersionStrategy

_BUILDER = bentoml.container.get_backend('buildx')
_module_location = pkg.source_locations('openllm')


def build_container(
  registries: LiteralContainerRegistry | t.Sequence[LiteralContainerRegistry] | None = None,
  version_strategy: LiteralContainerVersionStrategy = 'release',
  push: bool = False,
  machine: bool = False,
) -> dict[str | LiteralContainerRegistry, str]:
  try:
    if not _BUILDER.health():
      raise openllm.exceptions.Error
  except (openllm.exceptions.Error, subprocess.CalledProcessError):
    raise RuntimeError(
      'Building base container requires BuildKit (via Buildx) to be installed. See https://docs.docker.com/build/buildx/install/ for instalation instruction.'
    ) from None
  if not shutil.which('nvidia-container-runtime'):
    raise RuntimeError('NVIDIA Container Toolkit is required to compile CUDA kernel in container.')
  if not _module_location:
    raise RuntimeError("Failed to determine source location of 'openllm'. (Possible broken installation)")
  pyproject_path = pathlib.Path(_module_location).parent.parent / 'pyproject.toml'
  if not pyproject_path.exists():
    raise ValueError(
      "This utility can only be run within OpenLLM git repository. Clone it first with 'git clone https://github.com/bentoml/OpenLLM.git'"
    )
  if not registries:
    tags = {
      alias: openllm.bundle.RefResolver.construct_base_image(alias, version_strategy)
      for alias in openllm.bundle.CONTAINER_NAMES
    }
  else:
    registries = [registries] if isinstance(registries, str) else list(registries)
    tags = {name: openllm.bundle.RefResolver.construct_base_image(name, version_strategy) for name in registries}
  try:
    outputs = _BUILDER.build(
      file=pathlib.Path(__file__).parent.joinpath('Dockerfile').resolve().__fspath__(),
      context_path=pyproject_path.parent.__fspath__(),
      tag=tuple(tags.values()),
      push=push,
      progress='plain' if get_debug_mode() else 'auto',
      quiet=machine,
    )
    if machine and outputs is not None:
      tags['image_sha'] = outputs.decode('utf-8').strip()
  except Exception as err:
    raise openllm.exceptions.OpenLLMException(
      f'Failed to containerize base container images (Scroll up to see error above, or set DEBUG=5 for more traceback):\n{err}'
    ) from err
  return tags


@click.command(
  'build_base_container',
  context_settings=termui.CONTEXT_SETTINGS,
  help='''Base image builder for BentoLLM.

          By default, the base image will include custom kernels (PagedAttention via vllm, FlashAttention-v2, etc.) built with CUDA 11.8, Python 3.9 on Ubuntu22.04.
          Optionally, this can also be pushed directly to remote registry. Currently support ``docker.io``, ``ghcr.io`` and ``quay.io``.

          \b
          If '--machine' is passed, then it will run the process quietly, and output a JSON to the current running terminal.
          This command is only useful for debugging and for building custom base image for extending BentoML with custom base images and custom kernels.

          Note that we already release images on our CI to ECR and GHCR, so you don't need to build it yourself.
          ''',
)
@container_registry_option
@click.option(
  '--version-strategy',
  type=click.Choice(['release', 'latest', 'nightly']),
  default='nightly',
  help='Version strategy to use for tagging the image.',
)
@click.option('--push/--no-push', help='Whether to push to remote repository', is_flag=True, default=False)
@machine_option
def cli(
  container_registry: tuple[LiteralContainerRegistry, ...] | None,
  version_strategy: LiteralContainerVersionStrategy,
  push: bool,
  machine: bool,
) -> dict[str, str]:
  mapping = build_container(container_registry, version_strategy, push, machine)
  if machine:
    termui.echo(orjson.dumps(mapping, option=orjson.OPT_INDENT_2).decode(), fg='white')
  return mapping
