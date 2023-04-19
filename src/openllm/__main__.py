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

import typing as t

import click

import openllm

_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


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


@cli.group(cls=openllm.cli.StartCommand, context_settings=_CONTEXT_SETTINGS)
def start():
    """
    Start any LLM as a REST server.

    $ openllm start <model_name> --<options> ...
    """


@cli.group(cls=openllm.cli.StartCommand, context_settings=_CONTEXT_SETTINGS, _serve_grpc=True, name="start-grpc")
def start_grpc():
    """
    Start any LLM as a gRPC server.

    $ openllm start-grpc <model_name> --<options> ...
    """


@cli.command(aliases=["bundle"])
def build():
    """
    Package a given models.

    If given format is container, then also package the bundle into a container.
    """


@cli.command(hidden=True)
def deploy():
    """
    Deploy a model to a target platform.

    Deployment options:
    - BentoCloud
    - Self-hosted Yatai
    - SageMaker, ECR, EC2
    """


@cli.command(name="supported-models")
def supported_models():
    """
    List all supported models.
    """
    click.secho(f"\nSupported LLM: {', '.join(openllm.CONFIG_MAPPING.keys())}", fg="blue")


if __name__ == "__main__":
    cli()
