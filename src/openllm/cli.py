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

import functools
import inspect
import logging
import os
import sys
import time
import typing as t

import bentoml
import click
import inflection
import orjson
import rich.box
from bentoml_cli.utils import BentoMLCommandGroup
from click.utils import make_default_short_help
from click_option_group import optgroup
from rich.console import Console
from rich.table import Table

import openllm

if t.TYPE_CHECKING:
    P = t.ParamSpec("P")

    F = t.Callable[P, t.Any]

    class ClickFunctionProtocol(t.Protocol[P]):
        __name__: str
        __click_params__: list[click.Option]

        def __call__(*args: P.args, **attrs: P.kwargs) -> t.Any:
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

_console = Console()

output_decorator = click.option(
    "-o",
    "--output",
    type=click.Choice(["json", "pretty", "porcelain"]),
    default="pretty",
    help="Showing output type. Default to 'pretty'",
)


class OpenLLMCommandGroup(BentoMLCommandGroup):
    NUMBER_OF_COMMON_PARAMS = 3

    @staticmethod
    def common_params(f: F[P]) -> ClickFunctionProtocol[t.Any]:
        # The following logics is similar to one of BentoMLCommandGroup

        from bentoml._internal.configuration import DEBUG_ENV_VAR, QUIET_ENV_VAR, set_debug_mode, set_quiet_mode
        from bentoml._internal.log import configure_logging

        @click.option("-q", "--quiet", envvar=QUIET_ENV_VAR, is_flag=True, default=False, help="Suppress all output.")
        @click.option(
            "--debug", "--verbose", envvar=DEBUG_ENV_VAR, is_flag=True, default=False, help="Print out debug logs."
        )
        @click.option(
            "--do-not-track",
            is_flag=True,
            default=False,
            envvar=openllm.utils.analytics.OPENLLM_DO_NOT_TRACK,
            help="Do not send usage info",
        )
        @functools.wraps(f)
        def wrapper(quiet: bool, debug: bool, *args: P.args, **attrs: P.kwargs) -> t.Any:
            if quiet:
                set_quiet_mode(True)
                if debug:
                    logger.warning("'--quiet' passed; ignoring '--verbose/--debug'")
            elif debug:
                set_debug_mode(True)

            configure_logging()

            return f(*args, **attrs)

        return t.cast("ClickFunctionProtocol[t.Any]", wrapper)

    @staticmethod
    def usage_tracking(func: F[P], group: click.Group, **attrs: t.Any) -> ClickFunctionProtocol[t.Any]:
        from openllm.utils import analytics

        command_name = attrs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(do_not_track: bool, *args: P.args, **attrs: P.kwargs) -> t.Any:
            if do_not_track:
                with analytics.set_bentoml_tracking():
                    return func(*args, **attrs)

            start_time = time.time_ns()

            def get_tracking_event(return_value: t.Any):
                assert group.name, "Group name is required"
                if group.name in analytics.cli_events_map and command_name in analytics.cli_events_map[group.name]:
                    return analytics.cli_events_map[group.name][command_name](group, command_name, return_value)
                return analytics.OpenllmCliEvent(cmd_group=group.name, cmd_name=command_name)

            with analytics.set_bentoml_tracking():
                try:
                    return_value = func(*args, **attrs)
                    event = get_tracking_event(return_value)
                    duration_in_ms = (time.time_ns() - start_time) / 1e6
                    event.duration_in_ms = duration_in_ms
                    analytics.track(event)
                    return return_value
                except Exception as e:
                    event = get_tracking_event(None)
                    duration_in_ms = (time.time_ns() - start_time) / 1e6
                    event.duration_in_ms = duration_in_ms
                    event.error_type = type(e).__name__
                    event.return_code = 2 if isinstance(e, KeyboardInterrupt) else 1
                    analytics.track(event)
                    raise

        return t.cast("ClickFunctionProtocol[t.Any]", wrapper)

    @staticmethod
    def exception_handling(func: F[P], group: click.Group, **attrs: t.Any) -> ClickFunctionProtocol[t.Any]:
        command_name = attrs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **attrs: P.kwargs) -> t.Any:
            from bentoml._internal.configuration import get_debug_mode

            try:
                return func(*args, **attrs)
            except openllm.exceptions.OpenLLMException as err:
                if get_debug_mode():
                    _console.print_exception(show_locals=True, suppress=("click",))
                    raise
                raise click.ClickException(
                    click.style(f"[{group.name}] '{command_name}' failed: " + err.message, fg="red")
                ) from err
            except KeyboardInterrupt:  # NOTE: silience KeyboardInterrupt
                pass

        return t.cast("ClickFunctionProtocol[t.Any]", wrapper)

    def __init__(self, *args: t.Any, **attrs: t.Any) -> None:
        super(OpenLLMCommandGroup, self).__init__(*args, **attrs)
        # these two dictionaries will store known aliases for commands and groups
        self._alias_groups: dict[str, tuple[str, click.Group]] = {}

    def group(self, *args: t.Any, **attrs: t.Any) -> t.Callable[[F[P]], click.Group]:
        """Override the default 'cli.group' with supports for aliases for given group."""
        aliases = attrs.pop("aliases", [])

        def wrapper(f: F[P]) -> click.Group:
            from gettext import gettext as _

            name = attrs.pop("name", f.__name__)
            group = super(OpenLLMCommandGroup, self).group(name, *args, **attrs)(f)
            if group.short_help:
                text = inspect.cleandoc(group.short_help)
            elif group.help:
                text = make_default_short_help(group.help, 45)
            else:
                text = ""

            if group.deprecated:
                text = _("(Deprecated) {text}").format(text=text)

            attrs.setdefault("help", inspect.getdoc(f))
            attrs.setdefault("short_help", text)
            if len(aliases) > 0:
                for alias in aliases:
                    aliased_group = super(OpenLLMCommandGroup, self).group(alias, *args, **attrs)(f)
                    aliased_group.short_help = text + f" (alias for '{name}')"
                    self._alias_groups[name] = (alias, aliased_group)

            for __, g in self._alias_groups.values():
                g.commands = group.commands
            return group

        return wrapper

    def command(self, *args: t.Any, **attrs: t.Any) -> t.Callable[[F[P]], click.Command]:
        """Override the default 'cli.command' with supports for aliases for given command, and it
        wraps the implementation with common parameters.
        """
        if "context_settings" not in attrs:
            attrs["context_settings"] = {}
        attrs["context_settings"]["max_content_width"] = 120
        aliases = attrs.pop("aliases", None)

        def wrapper(f: F[P]) -> click.Command:
            name = f.__name__.lower().replace("_", "-")
            attrs.setdefault("help", inspect.getdoc(f))
            attrs.setdefault("name", name)

            # Wrap implementation withc common parameters
            wrapped = self.common_params(f)
            # Wrap into OpenLLM tracking
            wrapped = self.usage_tracking(wrapped, self, **attrs)
            # Wrap into exception handling
            wrapped = self.exception_handling(wrapped, self, **attrs)

            # move common parameters to end of the parameters list
            wrapped.__click_params__ = (
                wrapped.__click_params__[-self.NUMBER_OF_COMMON_PARAMS :]
                + wrapped.__click_params__[: -self.NUMBER_OF_COMMON_PARAMS]
            )

            # NOTE: we need to call super of super to avoid conflict with BentoMLCommandGroup command
            # setup
            cmd = super(BentoMLCommandGroup, self).command(*args, **attrs)(wrapped)
            # NOTE: add aliases to a given commands if it is specified.
            if aliases is not None:
                assert cmd.name
                self._commands[cmd.name] = aliases
                self._aliases.update({alias: cmd.name for alias in aliases})

            return cmd

        return wrapper


# NOTE: A list of bentoml option that is not needed for parsing.
# NOTE: User shouldn't set '--working-dir', as OpenLLM will setup this.
# NOTE: production is also deprecated
_IGNORED_OPTIONS = {"working_dir", "production", "protocol_version"}


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
        serve_options = [p for p in serve_command.params[1:-3] if p.name not in _IGNORED_OPTIONS]
        for options in reversed(serve_options):
            attrs = options.to_info_dict()
            # we don't need param_type_name, since it should all be options
            attrs.pop("param_type_name")
            # name is not a valid args
            attrs.pop("name")
            # type can be determine from default value
            attrs.pop("type")
            param_decls = (*attrs.pop("opts"), *attrs.pop("secondary_opts"))
            if name not in _IGNORED_OPTIONS:
                f = optgroup.option(*param_decls, **attrs)(f)

        return group(f)

    return decorator


def start_model_command(
    model_name: str,
    factory: click.Group,
    _context_settings: dict[str, t.Any] | None = None,
    _serve_grpc: bool = False,
) -> click.Command:
    """Generate a 'click.Command' for any given LLM.

    Args:
        model_name: The name of the model
        factory: The click.Group to add the command to
        _context_settings: The context settings to use for the command
        _serve_grpc: Whether to serve the model via gRPC or HTTP

    Returns:
        The click.Command for starting the model server

    Note that the internal commands will return the llm_config and a boolean determine
    whether the server is run with GPU or not.
    """
    envvar = openllm.utils.get_framework_env(model_name)
    model_command_decr: dict[str, t.Any] = {
        "name": inflection.underscore(model_name),
        "context_settings": _context_settings or {},
    }

    # TODO: Probably want to use docstring for the COMMAND_DOCSTRING here instead of just importing the module.
    config = openllm.AutoConfig.for_model(model_name)

    aliases: list[str] = []
    if config.__openllm_name_type__ == "dasherize":
        aliases.append(config.__openllm_start_name__)
    model_command_decr.update(
        {
            "name": config.__openllm_model_name__,
            "short_help": f"Start a LLMServer for '{model_name}' ('--help' for more details)",
            "help": getattr(
                openllm.utils.get_lazy_module(model_name),
                f"START_{inflection.underscore(model_name).upper()}_COMMAND_DOCSTRING",
            ),
            "aliases": aliases if len(aliases) > 0 else None,
        }
    )

    try:
        config.check_if_gpu_is_available(envvar)
    except openllm.exceptions.GpuNotAvailableError:
        # NOTE: The model requires GPU, therefore we will return a dummy command
        model_command_decr.update(
            {
                "short_help": "(Disabled because there is no GPU available)",
                "help": f"""{model_name} is currently not available to run on your 
                local machine because it requires GPU for faster inference.""",
            }
        )

        @factory.command(**model_command_decr)
        def noop() -> openllm.LLMConfig:
            click.secho("No GPU available, therefore this command is disabled", fg="red")
            openllm.utils.analytics.track_start_init(config, False)
            return config

        return noop

    @factory.command(**model_command_decr)
    @config.to_click_options
    @parse_serve_args(_serve_grpc)
    @click.option("--server-timeout", type=int, default=3600, help="Server timeout in seconds")
    @click.option(
        "--pretrained", type=click.STRING, default=None, help="Optional pretrained name or path to fine-tune weight."
    )
    def model_start(server_timeout: int, pretrained: str | None, **attrs: t.Any) -> openllm.LLMConfig:
        from bentoml._internal.configuration import get_debug_mode
        from bentoml._internal.log import configure_logging

        configure_logging()

        updated_config, server_kwds = config.model_validate_click(**attrs)
        openllm.utils.analytics.track_start_init(updated_config, False)

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
            + f' runners."llm-{config.__openllm_start_name__}-runner".timeout={config.__openllm_timeout__}'
        )

        start_env.update(
            {
                openllm.utils.FRAMEWORK_ENV_VAR(model_name): envvar,
                openllm.utils.MODEL_CONFIG_ENV_VAR(model_name): updated_config.model_dump_json(),
                "OPENLLM_MODEL": model_name,
                "BENTOML_DEBUG": str(get_debug_mode()),
                "BENTOML_CONFIG_OPTIONS": _bentoml_config_options,
            }
        )

        if envvar == "flax":
            llm = openllm.AutoFlaxLLM.for_model(model_name, pretrained=pretrained)
        elif envvar == "tf":
            llm = openllm.AutoTFLLM.for_model(model_name, pretrained=pretrained)
        else:
            llm = openllm.AutoLLM.for_model(model_name, pretrained=pretrained)

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

        # NOTE: Return the configuration for telemetry purposes.
        return updated_config

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
    start_model_command(model_name, t.cast(OpenLLMCommandGroup, cli), _serve_grpc=_serve_grpc)(
        standalone_mode=False, **attrs
    )


start = functools.partial(_start, _serve_grpc=False)
start_grpc = functools.partial(_start, _serve_grpc=True)


@click.group(cls=openllm.cli.OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="openllm")
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


@cli.command(name="version")
@output_decorator
def version(output: t.Literal["json", "pretty", "porcelain"]):
    """Return current OpenLLM version."""
    if output == "pretty":
        _console.print(f"OpenLLM version: {openllm.__version__}")
    elif output == "json":
        _console.print(orjson.dumps({"version": openllm.__version__}, option=orjson.OPT_INDENT_2).decode())
    else:
        click.echo(openllm.__version__)
    sys.exit(0)


@cli.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, aliases=["start-http"], name="start")
def start_cli():
    """
    Start any LLM as a REST server.

    $ openllm start <model_name> --<options> ...
    """


for name in openllm.CONFIG_MAPPING:
    start_cli.add_command(start_model_command(name, start_cli, _context_settings=_CONTEXT_SETTINGS))


@cli.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="start-grpc")
def start_grpc_cli():
    """
    Start any LLM as a gRPC server.

    $ openllm start-grpc <model_name> --<options> ...
    """


for name in openllm.CONFIG_MAPPING:
    start_grpc_cli.add_command(
        start_model_command(name, start_grpc_cli, _context_settings=_CONTEXT_SETTINGS, _serve_grpc=True)
    )


@cli.command(aliases=["build"])
@click.argument("model_name", type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()]))
@click.option("--pretrained", default=None, help="Given pretrained model name for the given model name [Optional].")
@click.option("--overwrite", is_flag=True, help="Overwrite existing Bento for given LLM if it already exists.")
@output_decorator
def bundle(model_name: str, pretrained: str | None, overwrite: bool, output: t.Literal["json", "pretty", "porcelain"]):
    """Package a given models into a Bento.

    $ openllm bundle flan-t5
    """
    from bentoml._internal.configuration import get_quiet_mode

    bento, _previously_built = openllm.build(
        model_name, __cli__=True, pretrained=pretrained, _overwrite_existing_bento=overwrite
    )

    if output == "pretty":
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
    elif output == "json":
        _console.print(orjson.dumps(bento.info.to_dict(), option=orjson.OPT_INDENT_2).decode())
    else:
        click.echo(bento.tag)
    return bento


@cli.command(name="models")
@output_decorator
def list_supported_models(output: t.Literal["json", "pretty", "porcelain"]):
    """
    List all supported models.
    """
    models = tuple(inflection.dasherize(key) for key in openllm.CONFIG_MAPPING.keys())
    if output == "pretty":
        table = Table(title="Supported LLMs", box=rich.box.SQUARE, show_lines=True)
        table.add_column("LLM")
        table.add_column("Description")
        table.add_column("Variants")
        for m in models:
            table.add_row(
                m,
                inspect.cleandoc(openllm.AutoConfig.for_model(m).__doc__ or "(No description)"),
                f"{openllm.AutoLLM.for_model(m).variants}",
            )
        _console.print(table)
    elif output == "json":
        _console.print(
            orjson.dumps(
                {
                    m: {
                        "variants": openllm.AutoLLM.for_model(m).variants,
                        "description": inspect.cleandoc(openllm.AutoConfig.for_model(m).__doc__ or "(No description)"),
                    }
                    for m in models
                },
                option=orjson.OPT_INDENT_2,
            ).decode()
        )
    else:
        click.echo("\n".join(models))
    sys.exit(0)


if __name__ == "__main__":
    cli()
