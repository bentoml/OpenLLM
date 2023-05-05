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
import subprocess
import threading
import typing as t

import bentoml
import click
from click_option_group import optgroup

import openllm

if t.TYPE_CHECKING:
    from openllm.types import F, P

    class ClickFunctionProtocol(t.Protocol[P]):
        __name__: str
        __click_params__: list[click.Option]

        def __call__(*args: P.args, **kwargs: P.kwargs) -> t.Any:
            ...

    ServeCommand = t.Literal["serve", "serve-grpc"]


logger = logging.getLogger(__name__)


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

        return wrapper

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
        super(OpenLLMCommandGroup, self).__init__(*args, **kwargs)
        # these two dictionaries will store known aliases for commands and groups
        self._commands: dict[str, list[str]] = {}
        self._aliases: dict[str, str] = {}

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

    def command(self, *args: t.Any, **kwargs: t.Any) -> t.Callable[[F[P]], click.Command]:
        if "context_settings" not in kwargs:
            kwargs["context_settings"] = {}
        kwargs["context_settings"]["max_content_width"] = 119
        aliases = kwargs.pop("aliases", None)

        def wrapper(f: F[P]) -> click.Command:
            name = f.__name__.lower().replace("_", "-")
            kwargs.setdefault("help", inspect.getdoc(f))
            kwargs.setdefault("name", name)

            cmd = super(OpenLLMCommandGroup, self).command(*args, **kwargs)(OpenLLMCommandGroup.common_chain(f))
            # add aliases to a given commands if it is specified.
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


class StartCommand(click.MultiCommand):
    def __init__(self, *args: t.Any, **kwargs: t.Any):
        self._serve_grpc = kwargs.pop("_serve_grpc", False)
        super(StartCommand, self).__init__(*args, **kwargs)
        self._cached_command: dict[str, click.Command] = {}

    def list_commands(self, ctx: click.Context):
        return openllm.CONFIG_MAPPING.keys()

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command:
        if cmd_name not in self._cached_command:
            self._cached_command[cmd_name] = start_model_command(cmd_name, _serve_grpc=self._serve_grpc)
        return self._cached_command[cmd_name]


def parse_serve_args(serve_grpc: bool) -> F[P]:
    """Parsing `bentoml serve|serve-grpc` click.Option to be parsed via `openllm start`"""
    from bentoml_cli.cli import cli

    command = "serve-http" if not serve_grpc else "serve-grpc"
    group = optgroup.group(
        f"Start a {'HTTP' if not serve_grpc else 'gRPC'} server options",
        help=f"Related to serving the model [synonymous to `bentoml {command}`]",
    )

    def decorator(f: F[P]) -> F[P]:
        _, serve_command, _ = cli.resolve_command(click.get_current_context(), [command])
        # The first variable is the argument bento
        # and the last three are shared default, which we don't need.
        serve_options = serve_command.params[1:-3]
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


def run_client(host: str, port: int, _serve_grpc: bool, llm_config_args: t.Any):
    import openllm_client

    client = openllm_client.create(f"{host}:{port}", kind="grpc" if _serve_grpc else "http", timeout=30)

    while True:
        try:
            assert client.health()
        except:
            continue
        else:
            break

    try:
        client.setup(**llm_config_args)
    except Exception as e:
        logger.error("Exception caught while setting up LLM Server:\n")
        logger.error(e)
        raise

    print("Successfully setup LLM Server")

    raise SystemExit(0)


def start_model_command(
    model_name: str,
    _context_settings: dict[str, t.Any] | None = None,
    _serve_grpc: bool = False,
) -> click.Command:
    config = openllm.Config.for_model(model_name)

    @click.command(
        model_name,
        short_help=f"Start a LLMServer for '{model_name}' ('--help' for more details)",
        context_settings=_context_settings or {},
        help=getattr(openllm, f"START_{openllm.utils.kebab_to_snake_case(model_name).upper()}_COMMAND_DOCSTRING"),
    )
    @openllm.LLMConfig.generate_click_options(config)
    @parse_serve_args(_serve_grpc)
    @openllm.cli.OpenLLMCommandGroup.common_chain
    def model_start(**attrs: t.Any):
        from bentoml._internal.log import configure_logging

        # NOTE: We need the below imports so that the client can use the custom IO Descriptor.
        from openllm.prompts import Prompt as Prompt

        configure_logging()

        start_env = {
            openllm.utils.FRAMEWORK_ENV_VAR(model_name): openllm.utils.get_framework_env(model_name),
        }

        llm_config_kwargs = {k: attrs[k] for k in config.__fields__ if k in attrs}
        # The rest should be server-related args
        server_args = {k: v for k, v in attrs.items() if k not in list(llm_config_kwargs.keys())}
        server_args.update(
            {
                "working_dir": openllm.utils.get_working_dir(model_name),
                "bento": f'service_{model_name.replace("-", "_")}:svc',
            }
        )
        # NOTE: currently, theres no development args in bentoml.Server. To be fixed upstream.
        development = server_args.pop("development")
        server_args.setdefault("production", not development)

        click.secho(
            f"\nStarting LLM Server for '{model_name}' at {'http://' if not _serve_grpc else ''}{server_args['host']}:{server_args['port']}\n",
            fg="blue",
        )
        server = getattr(bentoml, "HTTPServer" if not _serve_grpc else "GrpcServer")(**server_args)
        server.timeout = 90

        try:
            server.start(env=start_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            assert server.process is not None
            client = server.get_client()
            if llm_config_kwargs:
                click.secho(f"Setting default config for '{model_name}' with {llm_config_kwargs}", fg="blue")
                res = client.set_default_config(llm_config_kwargs)
                assert res

            with server.process.stdout:
                while True:
                    line = server.process.stdout.readline()
                    if not line:
                        break
                    click.secho(line.strip(), fg="blue")
        except KeyboardInterrupt:
            click.secho("\nStopping LLM Server...\n", fg="yellow")
            # TODO: Add possible next step
        except Exception as err:
            click.secho(f"Error caught while starting LLM Server:\n{err}", fg="red")
            raise

    return model_start
