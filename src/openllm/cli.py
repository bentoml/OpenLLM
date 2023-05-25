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
CLI utilities for OpenLLM.

This extends clidantic and BentoML's internal CLI CommandGroup.
"""
from __future__ import annotations

import difflib
import functools
import inspect
import logging
import os
import typing as t

import bentoml
import click
import inflection
import psutil
from click_option_group import optgroup

import openllm

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")

    F = t.Callable[P, t.Any]

    class ClickFunctionProtocol(t.Protocol[P]):
        __name__: str
        __click_params__: list[click.Option]

        def __call__(*args: P.args, **kwargs: P.kwargs) -> t.Any:
            ...

    ServeCommand = t.Literal["serve", "serve-grpc"]


logger = logging.getLogger(__name__)


_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": int(os.environ.get("COLUMNS", 120))}

OPENLLM_FIGLET = """\
 ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝
"""


class OpenLLMCommandGroup(click.Group):
    NUM_COMMON_PARAMS = 2

    @staticmethod
    def common_params(f: F[P]) -> ClickFunctionProtocol[t.Any]:
        # The following logics is similar to one of BentoMLCommandGroup

        from bentoml._internal.configuration import (DEBUG_ENV_VAR,
                                                     QUIET_ENV_VAR,
                                                     set_debug_mode,
                                                     set_quiet_mode)
        from bentoml._internal.log import configure_logging

        @click.option("-q", "--quiet", envvar=QUIET_ENV_VAR, is_flag=True, default=False, help="Suppress all output.")
        @click.option(
            "--debug", "--verbose", envvar=DEBUG_ENV_VAR, is_flag=True, default=False, help="Print out debug logs."
        )
        @functools.wraps(f)
        def wrapper(quiet: bool, debug: bool, *args: P.args, **kwargs: P.kwargs) -> t.Any:
            if quiet:
                set_quiet_mode(True)
                if debug:
                    logger.warning("'--quiet' passed; ignoring '--verbose/--debug'")
            elif debug:
                set_debug_mode(True)

            configure_logging()

            return f(*args, **kwargs)

        return t.cast("ClickFunctionProtocol[t.Any]", wrapper)

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super(OpenLLMCommandGroup, self).__init__(*args, **kwargs)
        # these two dictionaries will store known aliases for commands and groups
        self._commands: dict[str, list[str]] = {}
        self._aliases: dict[str, str] = {}
        self._alias_groups: dict[str, tuple[str, click.Group]] = {}

    # ported from bentoml_cli.utils.BentoMLCommandGroup to handle aliases and command difflib.
    def resolve_alias(self, cmd_name: str):
        return self._aliases[cmd_name] if cmd_name in self._aliases else cmd_name

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        cmd_name = self.resolve_alias(cmd_name)
        return super(OpenLLMCommandGroup, self).get_command(ctx, cmd_name)

    @staticmethod
    def common_chain(f: F[P]) -> ClickFunctionProtocol[t.Any]:
        # Wrap implementation withc common parameters
        wrapped = OpenLLMCommandGroup.common_params(f)
        # TODO: Tracking
        # TODO: Handling exception, using ExceptionGroup and Rich

        # move common parameters to end of the parameters list
        wrapped.__click_params__ = (
            wrapped.__click_params__[-OpenLLMCommandGroup.NUM_COMMON_PARAMS :]
            + wrapped.__click_params__[: -OpenLLMCommandGroup.NUM_COMMON_PARAMS]
        )
        return wrapped

    def group(self, *args: t.Any, **kwargs: t.Any) -> t.Callable[[F[P]], click.Group]:
        """Override the default 'cli.group' with supports for aliases for given group."""
        aliases = kwargs.pop("aliases", [])

        def wrapper(f: F[P]) -> click.Group:
            kwargs.setdefault("help", inspect.getdoc(f))
            if len(aliases) > 0:
                for alias in aliases:
                    aliased_group = super(OpenLLMCommandGroup, self).group(alias, *args, **kwargs)(f)
                    if aliased_group.short_help:
                        aliased_group.short_help += f" (alias for '{kwargs.get('name', f.__name__)}')"
                    else:
                        aliased_group.short_help = f"(alias for '{kwargs.get('name', f.__name__)}')"
                    self._alias_groups[kwargs.get("name", f.__name__)] = (alias, aliased_group)

            group = super(OpenLLMCommandGroup, self).group(*args, **kwargs)(f)
            for _, g in self._alias_groups.values():
                g.commands = group.commands
            return group

        return wrapper

    def command(self, *args: t.Any, **kwargs: t.Any) -> t.Callable[[F[P]], click.Command]:
        """Override the default 'cli.command' with supports for aliases for given command, and it
        wraps the implementation with common parameters.
        """
        if "context_settings" not in kwargs:
            kwargs["context_settings"] = {}
        kwargs["context_settings"]["max_content_width"] = 119
        aliases = kwargs.pop("aliases", None)

        def wrapper(f: F[P]) -> click.Command:
            name = f.__name__.lower().replace("_", "-")
            kwargs.setdefault("help", inspect.getdoc(f))
            kwargs.setdefault("name", name)

            cmd = super(OpenLLMCommandGroup, self).command(*args, **kwargs)(OpenLLMCommandGroup.common_chain(f))
            # NOTE: add aliases to a given commands if it is specified.
            if aliases is not None:
                assert cmd.name
                self._commands[cmd.name] = aliases
                self._aliases.update({alias: cmd.name for alias in aliases})

            return cmd

        return wrapper

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        rows: list[tuple[str, str]] = []
        sub_commands = self.list_commands(ctx)

        max_len = max(len(cmd) for cmd in sub_commands)
        limit = formatter.width - 6 - max_len

        for sub_command in sub_commands:
            cmd = self.get_command(ctx, sub_command)
            if cmd is None:
                continue
            # If the command is hidden, then we skip it.
            if hasattr(cmd, "hidden") and cmd.hidden:
                continue
            if sub_command in self._commands:
                aliases = ",".join(sorted(self._commands[sub_command]))
                sub_command = "%s (%s)" % (sub_command, aliases)
            # this cmd_help is available since click>=7
            # BentoML requires click>=7.
            cmd_help = cmd.get_short_help_str(limit)
            rows.append((sub_command, cmd_help))
        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)

    def resolve_command(
        self, ctx: click.Context, args: list[str]
    ) -> tuple[str | None, click.Command | None, list[str]]:
        try:
            return super(OpenLLMCommandGroup, self).resolve_command(ctx, args)
        except click.exceptions.UsageError as e:
            error_msg = str(e)
            original_cmd_name = click.utils.make_str(args[0])
            matches = difflib.get_close_matches(original_cmd_name, self.list_commands(ctx), 3, 0.5)
            if matches:
                fmt_matches = "\n    ".join(matches)
                error_msg += "\n\n"
                error_msg += f"Did you mean?\n    {fmt_matches}"
            raise click.exceptions.UsageError(error_msg, e.ctx)


class StartCommandGroup(OpenLLMCommandGroup):
    """A `start` factory that generate each models as its own click Command. See 'openllm start --help' for more details."""

    def __init__(self, *args: t.Any, **kwargs: t.Any):
        self._serve_grpc = kwargs.pop("_serve_grpc", False)
        self._cached_command: dict[str, click.Command] = {}
        super(StartCommandGroup, self).__init__(*args, **kwargs)

    def list_commands(self, ctx: click.Context):
        return openllm.CONFIG_MAPPING.keys()

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        cmd_name = self.resolve_alias(cmd_name)
        if cmd_name not in self._cached_command:
            self._cached_command[cmd_name] = start_model_command(cmd_name, self, _serve_grpc=self._serve_grpc)
        return self._cached_command[cmd_name]


def parse_serve_args(serve_grpc: bool) -> t.Callable[[F[P]], F[P]]:
    """Parsing `bentoml serve|serve-grpc` click.Option to be parsed via `openllm start`"""
    from bentoml_cli.cli import cli

    command = "serve" if not serve_grpc else "serve-grpc"
    group = optgroup.group(
        f"Start a {'HTTP' if not serve_grpc else 'gRPC'} server options",
        help=f"Related to serving the model [synonymous to `bentoml {'serve-http' if not serve_grpc else command }`]",
    )

    def decorator(f: F[P]) -> F[P]:
        serve_command = cli.commands[command]
        # The first variable is the argument bento
        # and the last three are shared default, which we don't need.
        serve_options = [p for p in serve_command.params[1:-3] if p.name not in ("protocol_version",)]
        for options in reversed(serve_options):
            attrs = options.to_info_dict()
            # we don't need param_type_name, since it should all be options
            attrs.pop("param_type_name")
            # name is not a valid args
            name = attrs.pop("name")
            # type can be determine from default value
            attrs.pop("type")
            param_decls = (*attrs.pop("opts"), *attrs.pop("secondary_opts"))
            # NOTE: User shouldn't set '--working-dir', as OpenLLM will setup this.
            # NOTE: production is also deprecated
            if name not in ("working_dir", "production"):
                f = optgroup.option(*param_decls, **attrs)(f)

        return group(f)

    return decorator


def start_model_command(
    model_name: str,
    factory: OpenLLMCommandGroup,
    _context_settings: dict[str, t.Any] | None = None,
    _serve_grpc: bool = False,
) -> click.Command:
    """Create a click.Command for starting a model server"""
    envvar = openllm.utils.get_framework_env(model_name)
    if envvar == "flax":
        llm = openllm.AutoFlaxLLM.for_model(model_name)
    elif envvar == "tf":
        llm = openllm.AutoTFLLM.for_model(model_name)
    else:
        llm = openllm.AutoLLM.for_model(model_name)

    aliases: list[str] = []
    if llm.config.__openllm_name_type__ == "dasherize":
        aliases.append(llm.config.__openllm_start_name__)

    @factory.command(
        name=llm.config.__openllm_model_name__,
        short_help=f"Start a LLMServer for '{model_name}' ('--help' for more details)",
        context_settings=_context_settings or {},
        help=getattr(
            openllm.utils.get_lazy_module(model_name),
            f"START_{inflection.underscore(model_name).upper()}_COMMAND_DOCSTRING",
        ),
        aliases=aliases if len(aliases) > 0 else None,
    )
    @llm.config.to_click_options
    @parse_serve_args(_serve_grpc)
    @click.option("--server-timeout", type=int, default=3600, help="Server timeout in seconds")
    def model_start(server_timeout: int, *args: t.Any, **attrs: t.Any):
        from bentoml._internal.configuration import get_debug_mode
        from bentoml._internal.log import configure_logging

        configure_logging()

        nw_config, server_kwds = llm.config.model_validate_click(**attrs)

        server_kwds.update({"working_dir": os.path.dirname(__file__)})
        if _serve_grpc:
            server_kwds["grpc_protocol_version"] = "v1"
        # NOTE: currently, theres no development args in bentoml.Server. To be fixed upstream.
        development = server_kwds.pop("development")
        server_kwds.setdefault("production", not development)

        start_env = os.environ.copy()

        # NOTE: This is a hack to set current configuration
        _bentoml_config_options = start_env.pop("BENTOML_CONFIG_OPTIONS", "")
        _bentoml_config_options += (
            " "
            if _bentoml_config_options
            else ""
            + f"api_server.timeout={server_timeout}"
            + f' runners."llm-{llm.config.__openllm_start_name__}-runner".timeout={llm.config.__openllm_timeout__}'
        )

        start_env.update(
            {
                openllm.utils.FRAMEWORK_ENV_VAR(model_name): envvar,
                openllm.utils.MODEL_CONFIG_ENV_VAR(model_name): nw_config.model_dump_json(),
                "OPENLLM_MODEL": model_name,
                "BENTOML_DEBUG": str(get_debug_mode()),
                "BENTOML_CONFIG_OPTIONS": _bentoml_config_options,
            }
        )

        if llm.requirements is not None:
            click.secho(
                f"Make sure that you have the following dependencies available: {llm.requirements}\n", fg="yellow"
            )
        click.secho(f"Starting LLM Server for '{model_name}'\n", fg="blue")
        server_cls = getattr(bentoml, "HTTPServer" if not _serve_grpc else "GrpcServer")
        server: bentoml.server.Server = server_cls("_service.py:svc", **server_kwds)
        server.timeout = 90

        try:
            server.start(env=start_env, text=True)
            assert server.process and server.process.stdout
            with server.process.stdout:
                for f in iter(server.process.stdout.readline, b""):
                    click.secho(f, fg="green", nl=False)
        except Exception as err:
            click.secho(f"Error caught while starting LLM Server:\n{err}", fg="red")
            raise
        finally:
            click.secho("\nStopping LLM Server...\n", fg="yellow")
            click.secho(
                f"Next step: you can run 'openllm bundle {model_name}' to create a Bento for {model_name}", fg="blue"
            )

    return model_start


def _start(
    model_name: str,
    framework: t.Literal["flax", "tf", "pt"] | None = None,
    **attrs: t.Any,
):
    """Python API to start a LLM server."""

    _serve_grpc = attrs.pop("_serve_grpc", False)

    if framework is not None:
        os.environ[openllm.utils.FRAMEWORK_ENV_VAR(model_name)] = framework
    start_model_command(model_name, cli, _serve_grpc=_serve_grpc)(standalone_mode=False, **attrs)


start = functools.partial(_start, _serve_grpc=False)
start_grpc = functools.partial(_start, _serve_grpc=True)


def create_cli() -> click.Group:
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

    @cli.group(
        cls=openllm.cli.StartCommandGroup, context_settings=_CONTEXT_SETTINGS, _serve_grpc=True, name="start-grpc"
    )
    def start_grpc():
        """
        Start any LLM as a gRPC server.

        $ openllm start-grpc <model_name> --<options> ...
        """

    @cli.command(aliases=["build"])
    @click.argument(
        "model_name", type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()])
    )
    @click.option("--pretrained", default=None, help="Given pretrained model name for the given model name [Optional].")
    @click.option("--overwrite", is_flag=True, help="Overwrite existing Bento for given LLM if it already exists.")
    def bundle(model_name: str, pretrained: str | None, overwrite: bool):
        """Package a given models into a Bento.

        $ openllm bundle flan-t5
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
                    f"'{model_name}' already has a Bento built [{bento}]. To overwrite it pass '--overwrite'.",
                    fg="yellow",
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

    if psutil.WINDOWS:
        import sys

        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore
    return cli


cli = create_cli()

if __name__ == "__main__":
    cli()
