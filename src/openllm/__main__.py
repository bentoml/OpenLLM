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
"""
CLI entrypoint for OpenLLM.

Usage:
    openllm --help

To start any OpenLLM model:
    openllm start <model_name> --options ...
"""
from __future__ import annotations

import os

import click
import inflection

import openllm

_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": int(os.environ.get("COLUMNS", 120))}

OPENLLM_FIGLET = """\
 ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝
"""


@click.group(cls=openllm.cli.OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS)
@click.version_option(openllm.__version__, "-v", "--version")
def cli():
    """
    \b
     ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
    ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
    ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
    ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
    ╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
     ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝

    \b
    OpenLLM: Your one stop-and-go-solution for serving any Open Large-Language Model

        - StableLM, Llama, Alpaca, Dolly, Flan-T5, and more

    \b
        - Powered by BentoML 🍱 + HuggingFace 🤗
    """


@cli.group(cls=openllm.cli.StartCommandGroup, context_settings=_CONTEXT_SETTINGS, aliases=["start-http"])
def start():
    """
    Start any LLM as a REST server.

    $ openllm start <model_name> --<options> ...
    """


@cli.group(cls=openllm.cli.StartCommandGroup, context_settings=_CONTEXT_SETTINGS, _serve_grpc=True, name="start-grpc")
def start_grpc():
    """
    Start any LLM as a gRPC server.

    $ openllm start-grpc <model_name> --<options> ...
    """


@cli.command(aliases=["build"])
@click.argument("model_name", type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()]))
@click.option("--pretrained", default=None, help="Given pretrained model name for the given model name [Optional].")
@click.option("--overwrite", is_flag=True, help="Overwrite existing Bento for given LLM if it already exists.")
def bundle(model_name: str, pretrained: str | None, overwrite: bool):
    """Package a given models into a Bento.

    $ openllm flan-t5
    """
    from bentoml._internal.configuration import get_quiet_mode

    bento, _previously_built = openllm.build(
        model_name, __cli__=True, pretrained=pretrained, _overwrite_existing_bento=overwrite
    )

    if not get_quiet_mode():
        click.echo("\n" + OPENLLM_FIGLET)
        if not _previously_built:
            click.secho(f"Successfully built {bento}.", fg="green")
        else:
            click.secho(
                f"'{model_name}' already has a Bento built [{bento}]. To overwrite it pass '--overwrite'.", fg="yellow"
            )

        click.secho(
            f"\nPossible next steps:\n\n * Push to BentoCloud with `bentoml push`:\n    $ bentoml push {bento.tag}",
            fg="blue",
        )
        click.secho(
            f"\n * Containerize your Bento with `bentoml containerize`:\n    $ bentoml containerize {bento.tag}",
            fg="blue",
        )


@cli.command(name="models")
def list_supported_models():
    """
    List all supported models.
    """
    click.secho(
        f"\nSupported LLM: [{', '.join(map(lambda key: inflection.dasherize(key), openllm.CONFIG_MAPPING.keys()))}]",
        fg="blue",
    )


if __name__ == "__main__":
    cli()
