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
import traceback
import typing as t

import bentoml
import click
import click_option_group as cog
import inflection
import orjson
import psutil
from bentoml._internal.configuration import get_debug_mode, get_quiet_mode
from bentoml_cli.utils import BentoMLCommandGroup

import openllm

if t.TYPE_CHECKING:
    from ._types import ClickFunctionWrapper, F, P

    ServeCommand = t.Literal["serve", "serve-grpc"]
    OutputLiteral = t.Literal["json", "pretty", "porcelain"]


logger = logging.getLogger(__name__)

COLUMNS = int(os.environ.get("COLUMNS", 120))

_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"], "max_content_width": COLUMNS}

OPENLLM_FIGLET = """\
 ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝
"""


def _echo(text: t.Any, fg: str = "green", _with_style: bool = True, **attrs: t.Any) -> None:
    call = click.echo
    if _with_style:
        attrs["fg"] = fg if not get_debug_mode() else None
        call = click.secho
    call(text, **attrs)


class OpenLLMCommandGroup(BentoMLCommandGroup):
    NUMBER_OF_COMMON_PARAMS = 3

    @staticmethod
    def common_params(f: F[P, t.Any]) -> ClickFunctionWrapper[..., t.Any]:
        """This is not supposed to be used with unprocessed click function.
        This should be used a the last currying from common_params -> usage_tracking -> exception_handling
        """
        # The following logics is similar to one of BentoMLCommandGroup

        from bentoml._internal.configuration import DEBUG_ENV_VAR, QUIET_ENV_VAR, set_debug_mode, set_quiet_mode
        from bentoml._internal.log import configure_logging

        from .utils import analytics

        @click.option("-q", "--quiet", envvar=QUIET_ENV_VAR, is_flag=True, default=False, help="Suppress all output.")
        @click.option(
            "--debug", "--verbose", envvar=DEBUG_ENV_VAR, is_flag=True, default=False, help="Print out debug logs."
        )
        @click.option(
            "--do-not-track",
            is_flag=True,
            default=False,
            envvar=analytics.OPENLLM_DO_NOT_TRACK,
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

        return t.cast("ClickFunctionWrapper[..., t.Any]", wrapper)

    @staticmethod
    def usage_tracking(
        func: ClickFunctionWrapper[..., t.Any], group: click.Group, **attrs: t.Any
    ) -> ClickFunctionWrapper[..., t.Any]:
        """This is not supposed to be used with unprocessed click function.
        This should be used a the last currying from common_params -> usage_tracking -> exception_handling
        """
        from .utils import analytics

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

        return t.cast("ClickFunctionWrapper[..., t.Any]", wrapper)

    @staticmethod
    def exception_handling(
        func: ClickFunctionWrapper[..., t.Any], group: click.Group, **attrs: t.Any
    ) -> ClickFunctionWrapper[..., t.Any]:
        """This is not supposed to be used with unprocessed click function.
        This should be used a the last currying from common_params -> usage_tracking -> exception_handling
        """
        command_name = attrs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(*args: P.args, **attrs: P.kwargs) -> t.Any:
            try:
                return func(*args, **attrs)
            except openllm.exceptions.OpenLLMException as err:
                raise click.ClickException(
                    click.style(f"[{group.name}] '{command_name}' failed: " + err.message, fg="red")
                ) from err
            except KeyboardInterrupt:  # NOTE: silience KeyboardInterrupt
                pass

        return t.cast("ClickFunctionWrapper[..., t.Any]", wrapper)

    def __init__(self, *args: t.Any, **attrs: t.Any) -> None:
        super(OpenLLMCommandGroup, self).__init__(*args, **attrs)
        # these two dictionaries will store known aliases for commands and groups
        self._cached_http: dict[str, t.Any] = {}
        self._cached_grpc: dict[str, t.Any] = {}

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        cmd_name = self.resolve_alias(cmd_name)
        if ctx.command.name == "start":
            if cmd_name not in self._cached_http:
                self._cached_http[cmd_name] = start_model_command(
                    cmd_name, self, _context_settings=ctx.command.context_settings
                )
            return self._cached_http[cmd_name]
        elif ctx.command.name == "start-grpc":
            if cmd_name not in self._cached_grpc:
                self._cached_grpc[cmd_name] = start_model_command(
                    cmd_name, self, _context_settings=ctx.command.context_settings, _serve_grpc=True
                )
            return self._cached_grpc[cmd_name]
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        if ctx.command.name == "start" or ctx.command.name == "start-grpc":
            return list(openllm.CONFIG_MAPPING.keys())

        return super().list_commands(ctx)

    def command(self, *args: t.Any, **attrs: t.Any) -> F[[t.Callable[P, t.Any]], click.Command]:
        """Override the default 'cli.command' with supports for aliases for given command, and it
        wraps the implementation with common parameters.
        """
        if "context_settings" not in attrs:
            attrs["context_settings"] = {}
        attrs["context_settings"]["max_content_width"] = 120
        aliases = attrs.pop("aliases", None)

        def wrapper(f: F[P, t.Any]) -> click.Command:
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

        # XXX: The current type coercion is not ideal, but we can really
        # loosely define it
        return t.cast("F[[t.Callable[..., t.Any]], click.Command]", wrapper)


# NOTE: A list of bentoml option that is not needed for parsing.
# NOTE: User shouldn't set '--working-dir', as OpenLLM will setup this.
# NOTE: production is also deprecated
_IGNORED_OPTIONS = {"working_dir", "production", "protocol_version"}


if t.TYPE_CHECKING:
    WrappedServeFunction = ClickFunctionWrapper[t.Concatenate[int, str | None, OutputLiteral, P], openllm.LLMConfig]
else:
    WrappedServeFunction = t.Any


def parse_serve_args(serve_grpc: bool):
    """Parsing `bentoml serve|serve-grpc` click.Option to be parsed via `openllm start`"""
    from bentoml_cli.cli import cli
    from click_option_group import optgroup

    command = "serve" if not serve_grpc else "serve-grpc"
    group = optgroup.group(
        f"Start a {'HTTP' if not serve_grpc else 'gRPC'} server options",
        help=f"Related to serving the model [synonymous to `bentoml {'serve-http' if not serve_grpc else command }`]",
    )

    def decorator(
        f: t.Callable[t.Concatenate[int, str | None, t.Literal["porcelain", "pretty"], P], openllm.LLMConfig]
    ) -> ClickFunctionWrapper[P, openllm.LLMConfig]:
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
            f = t.cast("WrappedServeFunction[P]", optgroup.option(*param_decls, **attrs)(f))

        return group(f)

    return decorator


def start_model_command(
    model_name: str,
    group: click.Group,
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
    from bentoml._internal.configuration.containers import BentoMLContainer
    from bentoml._internal.log import configure_logging

    configure_logging()

    ModelEnv = openllm.utils.ModelEnv(model_name)
    command_attrs: dict[str, t.Any] = {
        "name": ModelEnv.model_name,
        "context_settings": _context_settings or {},
        "short_help": f"Start a LLMServer for '{model_name}' ('--help' for more details)",
        "help": ModelEnv.start_docstring,
    }

    llm_config = openllm.AutoConfig.for_model(model_name)

    aliases: list[str] = []
    if llm_config.name_type == "dasherize":
        aliases.append(llm_config.__openllm_start_name__)

    command_attrs["aliases"] = aliases if len(aliases) > 0 else None

    gpu_available = False
    try:
        llm_config.check_if_gpu_is_available(ModelEnv.get_framework_env())
        gpu_available = True if llm_config.__openllm_requires_gpu__ else False
    except openllm.exceptions.GpuNotAvailableError:
        # NOTE: The model requires GPU, therefore we will return a dummy command
        command_attrs.update(
            {
                "short_help": "(Disabled because there is no GPU available)",
                "help": f"""{model_name} is currently not available to run on your
                local machine because it requires GPU for faster inference.""",
            }
        )

        @group.command(**command_attrs)
        def noop() -> openllm.LLMConfig:
            _echo("No GPU available, therefore this command is disabled", fg="red")
            openllm.utils.analytics.track_start_init(llm_config, gpu_available)
            return llm_config

        return noop

    @group.command(**command_attrs)
    @llm_config.to_click_options
    @parse_serve_args(_serve_grpc)
    @click.option("--server-timeout", type=int, default=3600, help="Server timeout in seconds")
    @click.option(
        "--pretrained", type=click.STRING, default=None, help="Optional pretrained name or path to fine-tune weight."
    )
    @click.option(
        "-o",
        "--output",
        type=click.Choice(["pretty", "porcelain"]),
        default="porcelain",
        help="Showing output type. Default to 'pretty'",
    )
    def model_start(
        server_timeout: int,
        pretrained: str | None,
        output: t.Literal["pretty", "porcelain"],
        **attrs: t.Any,
    ) -> openllm.LLMConfig:
        config, server_attrs = llm_config.model_validate_click(**attrs)

        if ModelEnv.get_framework_env() == "flax":
            llm = openllm.AutoFlaxLLM.for_model(model_name, pretrained=pretrained, llm_config=config)
        elif ModelEnv.get_framework_env() == "tf":
            llm = openllm.AutoTFLLM.for_model(model_name, pretrained=pretrained, llm_config=config)
        else:
            llm = openllm.AutoLLM.for_model(model_name, pretrained=pretrained, llm_config=config)

        # NOTE: We need to initialize llm here first to check if the model is already downloaded to
        # avoid deadlock before the subprocess forking.
        llm.ensure_pretrained_exists()

        # NOTE: check for GPU one more time in cases this model doesn't requires GPU but users can still
        # run this model on GPU
        try:
            llm.config.check_if_gpu_is_available(ModelEnv.get_framework_env(), force=True)
            gpu_available = True
        except openllm.exceptions.GpuNotAvailableError:
            gpu_available = False

        server_attrs.update({"working_dir": os.path.dirname(__file__)})
        if _serve_grpc:
            server_attrs["grpc_protocol_version"] = "v1"
        # NOTE: currently, theres no development args in bentoml.Server. To be fixed upstream.
        development = server_attrs.pop("development")
        server_attrs.setdefault("production", not development)

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
                ModelEnv.framework: ModelEnv.get_framework_env(),
                ModelEnv.model_config: llm.config.model_dump_json().decode(),
                "OPENLLM_MODEL": model_name,
                "BENTOML_DEBUG": str(get_debug_mode()),
                "BENTOML_CONFIG_OPTIONS": _bentoml_config_options,
                "BENTOML_HOME": os.environ.get("BENTOML_HOME", BentoMLContainer.bentoml_home.get()),
            }
        )

        if llm.requirements is not None:
            _echo(f"Make sure to have the following dependencies available: {llm.requirements}", fg="yellow")

        if t.TYPE_CHECKING:
            server_cls: type[bentoml.HTTPServer] if not _serve_grpc else type[bentoml.GrpcServer]

        server_cls = getattr(bentoml, "HTTPServer" if not _serve_grpc else "GrpcServer")
        server_attrs["timeout"] = 90
        server = server_cls("_service.py:svc", **server_attrs)

        try:
            openllm.utils.analytics.track_start_init(llm.config, gpu_available)
            server.start(env=start_env, text=True, blocking=True if get_debug_mode() else False)
            if not get_debug_mode():
                assert server.process is not None and server.process.stdout is not None
                with server.process.stdout:
                    for f in iter(server.process.stdout.readline, b""):
                        _echo(f, nl=False, fg="white")
        except KeyboardInterrupt:
            on_start_end(model_name)
        except Exception as err:
            _echo(f"Error caught while starting LLM Server:\n{err}", fg="red")
            raise
        else:
            on_start_end(model_name)

        # NOTE: Return the configuration for telemetry purposes.
        return llm_config

    def on_start_end(model_name: str):
        if not get_debug_mode():
            _echo(
                f"\n🚀 Next step: run 'openllm build {model_name}' to create a Bento for {model_name}",
                fg="blue",
            )

    return model_start


def _start(
    model_name: str,
    framework: t.Literal["flax", "tf", "pt"] | None = None,
    **attrs: t.Any,
):
    """Python API to start a LLM server."""
    from . import utils

    _serve_grpc = attrs.pop("_serve_grpc", False)

    ModelEnv = utils.ModelEnv(model_name)

    if framework is not None:
        os.environ[ModelEnv.framework] = framework
    start_model_command(model_name, t.cast(OpenLLMCommandGroup, cli), _serve_grpc=_serve_grpc)(
        standalone_mode=False, **attrs
    )


start = functools.partial(_start, _serve_grpc=False)
start_grpc = functools.partial(_start, _serve_grpc=True)


output_option = click.option(
    "-o",
    "--output",
    type=click.Choice(["json", "pretty", "porcelain"]),
    default="pretty",
    help="Showing output type. Default to 'pretty'",
)


def cli_factory() -> click.Group:
    from bentoml._internal.log import configure_logging

    configure_logging()

    @click.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="openllm")
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

            - StableLM, Falcon, ChatGLM, Dolly, Flan-T5, and more

        \b
            - Powered by BentoML 🍱
        """

    @cli.command()
    @output_option
    @click.pass_context
    def version(ctx: click.Context, output: OutputLiteral):
        """🚀 OpenLLM version."""
        from gettext import gettext

        from .__about__ import __version__

        message = gettext("%(prog)s, version %(version)s")
        prog_name = ctx.find_root().info_name

        if output == "pretty":
            _echo(message % {"prog": prog_name, "version": __version__}, color=ctx.color)
        elif output == "json":
            _echo(orjson.dumps({"version": __version__}, option=orjson.OPT_INDENT_2).decode())
        else:
            _echo(__version__)

        ctx.exit()

    @cli.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS)
    def start():
        """
        Start any LLM as a REST server.

        $ openllm start <model_name> --<options> ...
        """

    @cli.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS)
    def start_grpc():
        """
        Start any LLM as a gRPC server.

        $ openllm start-grpc <model_name> --<options> ...
        """

    @cli.command()
    @click.argument(
        "model_name", type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()])
    )
    @click.option("--pretrained", default=None, help="Given pretrained model name for the given model name [Optional].")
    @click.option("--overwrite", is_flag=True, help="Overwrite existing Bento for given LLM if it already exists.")
    @output_option
    def build(model_name: str, pretrained: str | None, overwrite: bool, output: OutputLiteral):
        """Package a given models into a Bento.

        $ openllm build flan-t5
        """
        bento, _previously_built = openllm.build(
            model_name,
            __cli__=True,
            pretrained=pretrained,
            _overwrite_existing_bento=overwrite,
        )

        if output == "pretty":
            if not get_quiet_mode():
                _echo("\n" + OPENLLM_FIGLET)
                if not _previously_built:
                    _echo(f"Successfully built {bento}.", fg="green")
                else:
                    _echo(
                        f"'{model_name}' already has a Bento built [{bento}]. To overwrite it pass '--overwrite'.",
                        fg="yellow",
                    )

                _echo(
                    "\nPossible next steps:\n\n"
                    + "* Push to BentoCloud with `bentoml push`:\n"
                    + f"    $ bentoml push {bento.tag}\n"
                    + "* Containerize your Bento with `bentoml containerize`:\n"
                    + f"    $ bentoml containerize {bento.tag}",
                    fg="blue",
                )
        elif output == "json":
            _echo(orjson.dumps(bento.info.to_dict(), option=orjson.OPT_INDENT_2).decode())
        else:
            _echo(bento.tag)
        return bento

    @cli.command(aliases=["list"])
    @output_option
    def models(output: OutputLiteral):
        """List all supported models."""
        models = tuple(inflection.dasherize(key) for key in openllm.CONFIG_MAPPING.keys())
        if output == "porcelain":
            _echo("\n".join(models), fg="white")
        else:
            failed_initialized: list[tuple[str, Exception]] = []

            json_data: dict[str, dict[t.Literal["variants", "description"], t.Any]] = {}

            for m in models:
                try:
                    model = openllm.AutoLLM.for_model(m)
                    docs = inspect.cleandoc(model.config.__doc__ or "(No description)")
                    json_data[m] = {"variants": model.variants, "description": docs}
                except Exception as err:
                    failed_initialized.append((m, err))

            if output == "pretty":
                import tabulate

                tabulate.PRESERVE_WHITESPACE = True

                data: list[str | tuple[str, str, list[str]]] = []
                for m, v in json_data.items():
                    data.extend([(m, v["description"], v["variants"])])
                column_widths = [int(COLUMNS / 6), int(COLUMNS / 3 * 2), int(COLUMNS / 6)]

                if len(data) == 0 and len(failed_initialized) > 0:
                    _echo("Exception found while parsing models:\n", fg="yellow")
                    for m, err in failed_initialized:
                        _echo(f"- {m}: ", fg="yellow", nl=False)
                        _echo(traceback.print_exception(err, limit=3), fg="red")
                    sys.exit(1)

                table = tabulate.tabulate(
                    data,
                    tablefmt="fancy_grid",
                    headers=["LLM", "Description", "Variants"],
                    maxcolwidths=column_widths,
                )

                formatted_table = ""
                for line in table.split("\n"):
                    formatted_table += (
                        "".join(f"{cell:{width}}" for cell, width in zip(line.split("\t"), column_widths)) + "\n"
                    )
                _echo(formatted_table, fg="white")

                if len(failed_initialized) > 0:
                    _echo("\nThe following models are supported but failed to initialize:\n")
                    for m, err in failed_initialized:
                        _echo(f"- {m}: ", fg="blue", nl=False)
                        _echo(err, fg="red")
            else:
                _echo(orjson.dumps(json_data, option=orjson.OPT_INDENT_2).decode())

        sys.exit(0)

    @cli.command(aliases=["save"])
    @click.argument(
        "model_name", type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()])
    )
    @click.option(
        "--pretrained", type=click.STRING, default=None, help="Optional pretrained name or path to fine-tune weight."
    )
    @output_option
    def download_models(model_name: str, pretrained: str | None, output: OutputLiteral):
        """Setup LLM interactively.

        Note: This is useful for development and setup for fine-tune.
        """
        config = openllm.AutoConfig.for_model(model_name)
        env = config.__openllm_env__.get_framework_env()
        if env == "flax":
            model = openllm.AutoFlaxLLM.for_model(model_name, pretrained=pretrained, llm_config=config)
        elif env == "tf":
            model = openllm.AutoTFLLM.for_model(model_name, pretrained=pretrained, llm_config=config)
        else:
            model = openllm.AutoLLM.for_model(model_name, pretrained=pretrained, llm_config=config)

        tag = model.make_tag()

        if len(bentoml.models.list(tag)) == 0:
            if output == "pretty":
                _echo(f"{tag} does not exists yet!. Downloading...", nl=True)
                m = model.ensure_pretrained_exists()
                _echo(f"Saved model: {m.tag}")
            elif output == "json":
                m = model.ensure_pretrained_exists()
                _echo(
                    orjson.dumps(
                        {"previously_setup": False, "framework": env, "tag": str(m.tag)}, option=orjson.OPT_INDENT_2
                    ).decode()
                )
            else:
                m = model.ensure_pretrained_exists()
                _echo(tag)
        else:
            m = model.ensure_pretrained_exists()
            if output == "pretty":
                _echo(f"{model_name} is already setup for framework '{env}': {str(m.tag)}", nl=True)
            elif output == "json":
                _echo(
                    orjson.dumps(
                        {"previously_setup": True, "framework": env, "model": str(m.tag)}, option=orjson.OPT_INDENT_2
                    ).decode()
                )
            else:
                _echo(m.tag)
        return m

    @cli.command(name="query", aliases=["run", "ask"])
    @cog.optgroup.group(
        "Host options", cls=cog.RequiredMutuallyExclusiveOptionGroup, help="default host for the running LLM server"
    )
    @cog.optgroup.option("--endpoint", type=click.STRING, help="LLM Server endpoint, i.e: http://12.323.2.1")
    @cog.optgroup.option("--local", type=click.BOOL, help="Whether the server is running locally.")
    @click.option("--port", type=click.INT, default=3000, help="LLM Server port", show_default=True)
    @click.option("--timeout", type=click.INT, default=30, help="Default server timeout", show_default=True)
    @click.option(
        "--server-type", type=click.Choice(["grpc", "http"]), help="Server type", default="http", show_default=True
    )
    @output_option
    @click.argument("query", type=click.STRING)
    def query(
        query: str,
        endpoint: str,
        port: int,
        timeout: int,
        local: bool,
        server_type: t.Literal["http", "grpc"],
        output: OutputLiteral,
    ):
        """Ask a LLM interactively, from a terminal.

        $ openllm query --endpoint http://12.323.2.1 "What is the meaning of life?"
        """
        target_url = f"http://0.0.0.0:{port}" if local else f"{endpoint}:{port}"

        client = (
            openllm.client.HTTPClient(target_url, timeout=timeout)
            if server_type == "http"
            else openllm.client.GrpcClient(target_url, timeout=timeout)
        )

        if client.framework == "flax":
            model = openllm.AutoFlaxLLM.for_model(client.model_name)
        elif client.framework == "tf":
            model = openllm.AutoTFLLM.for_model(client.model_name)
        else:
            model = openllm.AutoLLM.for_model(client.model_name)

        if output != "porcelain":
            _echo(f"Processing query: {query}\n", fg="white")

        res = client.query(query, return_raw_response=True)

        if output == "pretty":
            formatted = model.postprocess_generate(query, res["responses"])
            _echo("Responses: ", fg="white", nl=False)
            _echo(formatted, fg="cyan")
        elif output == "json":
            _echo(orjson.dumps(res, option=orjson.OPT_INDENT_2).decode(), fg="white")
        else:
            _echo(res["responses"], fg="white")

    if t.TYPE_CHECKING:
        assert download_models and build and models and version and start and start_grpc and query

    if psutil.WINDOWS:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    return cli


cli = cli_factory()

if __name__ == "__main__":
    cli()
