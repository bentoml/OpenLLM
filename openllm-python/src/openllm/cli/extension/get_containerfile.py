from __future__ import annotations
import typing as t

import click

from simple_di import Provide
from simple_di import inject

import bentoml

from bentoml._internal.bento.bento import BentoInfo
from bentoml._internal.bento.build_config import DockerOptions
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.container.generate import generate_containerfile
from openllm.cli import termui
from openllm.cli._factory import bento_complete_envvar
from openllm_core.utils import bentoml_cattr

if t.TYPE_CHECKING:
  from bentoml._internal.bento import BentoStore

@click.command('get_containerfile', context_settings=termui.CONTEXT_SETTINGS, help='Return Containerfile of any given Bento.')
@click.argument('bento', type=str, shell_complete=bento_complete_envvar)
@click.pass_context
@inject
def cli(ctx: click.Context, bento: str, _bento_store: BentoStore = Provide[BentoMLContainer.bento_store]) -> str:
  try:
    bentomodel = _bento_store.get(bento)
  except bentoml.exceptions.NotFound:
    ctx.fail(f'Bento {bento} not found. Make sure to call `openllm build` first.')
  # The logic below are similar to bentoml._internal.container.construct_containerfile
  with open(bentomodel.path_of('bento.yaml'), 'r') as f:
    options = BentoInfo.from_yaml_file(f)
    # NOTE: dockerfile_template is already included in the
    # Dockerfile inside bento, and it is not relevant to
    # construct_containerfile. Hence it is safe to set it to None here.
    # See https://github.com/bentoml/BentoML/issues/3399.
    docker_attrs = bentoml_cattr.unstructure(options.docker)
    # NOTE: if users specify a dockerfile_template, we will
    # save it to /env/docker/Dockerfile.template. This is necessary
    # for the reconstruction of the Dockerfile.
    if 'dockerfile_template' in docker_attrs and docker_attrs['dockerfile_template'] is not None:
      docker_attrs['dockerfile_template'] = 'env/docker/Dockerfile.template'
    doc = generate_containerfile(docker=DockerOptions(**docker_attrs), build_ctx=bentomodel.path, conda=options.conda, bento_fs=bentomodel._fs, enable_buildkit=True, add_header=True)
    termui.echo(doc, fg='white')
  return bentomodel.path
