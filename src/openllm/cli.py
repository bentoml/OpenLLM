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
import psutil
from bentoml_cli.utils import BentoMLCommandGroup

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


class OpenLLMCommandGroup(BentoMLCommandGroup):
    NUMBER_OF_COMMON_PARAMS = 3

    @staticmethod
    def common_params(f: F[P]) -> ClickFunctionProtocol[t.Any]:
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

        return t.cast("ClickFunctionProtocol[t.Any]", wrapper)

    @staticmethod
    def usage_tracking(func: F[P], group: click.Group, **attrs: t.Any) -> ClickFunctionProtocol[t.Any]:
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

        return t.cast("ClickFunctionProtocol[t.Any]", wrapper)

    @staticmethod
    def exception_handling(func: F[P], group: click.Group, **attrs: t.Any) -> ClickFunctionProtocol[t.Any]:
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

        return t.cast("ClickFunctionProtocol[t.Any]", wrapper)

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
    from click_option_group import optgroup

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
            f = optgroup.option(*param_decls, **attrs)(f)

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
    from bentoml._internal.configuration import get_debug_mode
    from bentoml._internal.log import configure_logging

    configure_logging()

    ModelEnv = openllm.utils.ModelEnv(model_name)
    model_command_decr: dict[str, t.Any] = {"name": ModelEnv.model_name, "context_settings": _context_settings or {}}

    # TODO: Probably want to use docstring for the COMMAND_DOCSTRING here instead of just importing the module.
    config = openllm.AutoConfig.for_model(model_name)

    aliases: list[str] = []
    if config.name_type == "dasherize":
        aliases.append(config.__openllm_start_name__)
    model_command_decr.update(
        {
            "name": config.__openllm_model_name__,
            "short_help": f"Start a LLMServer for '{model_name}' ('--help' for more details)",
            "help": ModelEnv.start_docstring,
            "aliases": aliases if len(aliases) > 0 else None,
        }
    )

    gpu_available = False
    try:
        config.check_if_gpu_is_available(ModelEnv.get_framework_env())
        gpu_available = True
    except openllm.exceptions.GpuNotAvailableError:
        # NOTE: The model requires GPU, therefore we will return a dummy command
        model_command_decr.update(
            {
                "short_help": "(Disabled because there is no GPU available)",
                "help": f"""{model_name} is currently not available to run on your 
                local machine because it requires GPU for faster inference.""",
            }
        )

        @group.command(**model_command_decr)
        def noop() -> openllm.LLMConfig:
            click.secho("No GPU available, therefore this command is disabled", fg="red")
            openllm.utils.analytics.track_start_init(config, gpu_available)
            return config

        return noop

    @group.command(**model_command_decr)
    @config.to_click_options
    @parse_serve_args(_serve_grpc)
    @click.option("--server-timeout", type=int, default=3600, help="Server timeout in seconds")
    @click.option(
        "--pretrained", type=click.STRING, default=None, help="Optional pretrained name or path to fine-tune weight."
    )
    def model_start(server_timeout: int, pretrained: str | None, **attrs: t.Any) -> openllm.LLMConfig:
        from bentoml._internal.configuration.containers import BentoMLContainer

        nonlocal config
        config, server_attrs = config.model_validate_click(**attrs)

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
            llm.config.check_if_gpu_is_available(ModelEnv.get_framework_env())
            gpu_available = True
        except openllm.exceptions.GpuNotAvailableError:
            gpu_available = False

        openllm.utils.analytics.track_start_init(llm.config, gpu_available)

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
                ModelEnv.model_config: llm.config.model_dump_json(),
                "OPENLLM_MODEL": model_name,
                "BENTOML_DEBUG": str(get_debug_mode()),
                "BENTOML_CONFIG_OPTIONS": _bentoml_config_options,
                "BENTOML_HOME": os.environ.get("BENTOML_HOME", BentoMLContainer.bentoml_home.get()),
            }
        )

        if llm.requirements is not None:
            click.secho(
                f"Make sure that you have the following dependencies available: {llm.requirements}\n", fg="yellow"
            )
        click.secho(f"\nStarting LLM Server for '{model_name}'\n", fg="blue")
        server_cls = getattr(bentoml, "HTTPServer" if not _serve_grpc else "GrpcServer")
        server: bentoml.server.Server = server_cls("_service.py:svc", **server_attrs)
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
        return config

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


def create_cli() -> click.Group:
    output_decorator = click.option(
        "-o",
        "--output",
        type=click.Choice(["json", "pretty", "porcelain"]),
        default="pretty",
        help="Showing output type. Default to 'pretty'",
    )

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

            - StableLM, Llama, Alpaca, Dolly, Flan-T5, and more

        \b
            - Powered by BentoML 🍱 + HuggingFace 🤗
        """

    @cli.command(name="version")
    @output_decorator
    @click.pass_context
    def _(ctx: click.Context, output: t.Literal["json", "pretty", "porcelain"]):
        """🚀 OpenLLM version."""
        from gettext import gettext

        from .__about__ import __version__

        message = gettext("%(prog)s, version %(version)s")
        version = __version__
        prog_name = ctx.find_root().info_name

        if output == "pretty":
            click.echo(message % {"prog": prog_name, "version": version}, color=ctx.color)
        elif output == "json":
            click.echo(orjson.dumps({"version": version}, option=orjson.OPT_INDENT_2).decode())
        else:
            click.echo(version)

        ctx.exit()

    @cli.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="start")
    def _():
        """
        Start any LLM as a REST server.

        $ openllm start <model_name> --<options> ...
        """

    @cli.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="start-grpc")
    def _():
        """
        Start any LLM as a gRPC server.

        $ openllm start-grpc <model_name> --<options> ...
        """

    @cli.command(name="bundle", aliases=["build"])
    @click.argument(
        "model_name", type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()])
    )
    @click.option("--pretrained", default=None, help="Given pretrained model name for the given model name [Optional].")
    @click.option("--overwrite", is_flag=True, help="Overwrite existing Bento for given LLM if it already exists.")
    @output_decorator
    def _(model_name: str, pretrained: str | None, overwrite: bool, output: t.Literal["json", "pretty", "porcelain"]):
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
                    "\nPossible next steps:\n\n * Push to BentoCloud with `bentoml push`:\n    "
                    + f"$ bentoml push {bento.tag}",
                    fg="blue",
                )
                click.secho(
                    "\n * Containerize your Bento with `bentoml containerize`:\n    "
                    + f"$ bentoml containerize {bento.tag}",
                    fg="blue",
                )
        elif output == "json":
            click.secho(orjson.dumps(bento.info.to_dict(), option=orjson.OPT_INDENT_2).decode())
        else:
            click.echo(bento.tag)
        return bento

    @cli.command(name="models")
    @output_decorator
    def _(output: t.Literal["json", "pretty", "porcelain"]):
        """List all supported models."""
        models = tuple(inflection.dasherize(key) for key in openllm.CONFIG_MAPPING.keys())
        failed_initialized: list[tuple[str, Exception]] = []
        if output == "pretty":
            import rich
            import rich.box
            from rich.table import Table
            from rich.text import Text

            console = rich.get_console()
            table = Table(title="Supported LLMs", box=rich.box.SQUARE, show_lines=True)
            table.add_column("LLM")
            table.add_column("Description")
            table.add_column("Variants")
            for m in models:
                docs = inspect.cleandoc(openllm.AutoConfig.for_model(m).__doc__ or "(No description)")
                try:
                    model = openllm.AutoLLM.for_model(m)
                    table.add_row(m, docs, f"{model.variants}")
                except Exception as err:
                    failed_initialized.append((m, err))
            console.print(table)
            if len(failed_initialized) > 0:
                console.print(
                    "\n[bold yellow] The following models are supported but failed to initialize:[/bold yellow]\n"
                )
                for m, err in failed_initialized:
                    console.print(Text(f"- {m}: ") + Text(f"{err}\n", style="bold red"))
        elif output == "json":
            result_json: dict[str, dict[t.Literal["variants", "description"], t.Any]] = {}
            for m in models:
                docs = inspect.cleandoc(openllm.AutoConfig.for_model(m).__doc__ or "(No description)")
                try:
                    model = openllm.AutoLLM.for_model(m)
                    result_json[m] = {"variants": model.variants, "description": docs}
                except Exception as err:
                    logger.debug("Exception caught while parsing model %s", m, exc_info=err)
                    result_json[m] = {"variants": None, "description": docs}

            click.secho(orjson.dumps(result_json, option=orjson.OPT_INDENT_2).decode())
        else:
            click.echo("\n".join(models))
        sys.exit(0)

    @cli.command(name="download-models")
    @click.argument(
        "model_name", type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()])
    )
    @click.option(
        "--pretrained", type=click.STRING, default=None, help="Optional pretrained name or path to fine-tune weight."
    )
    @output_decorator
    def _(model_name: str, pretrained: str | None, output: t.Literal["json", "pretty", "porcelain"]):
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
                click.secho(f"{tag} does not exists yet!. Downloading...", nl=True)
                m = model.ensure_pretrained_exists()
                click.secho(f"Saved model: {m.tag}")
            elif output == "json":
                m = model.ensure_pretrained_exists()
                click.secho(
                    orjson.dumps(
                        {"previously_setup": False, "framework": env, "tag": str(m.tag)}, option=orjson.OPT_INDENT_2
                    ).decode()
                )
            else:
                m = model.ensure_pretrained_exists()
                click.secho(m.tag)
        else:
            m = model.ensure_pretrained_exists()
            if output == "pretty":
                click.secho(f"{model_name} is already setup for framework '{env}': {str(m.tag)}", nl=True)
            elif output == "json":
                click.secho(
                    orjson.dumps(
                        {"previously_setup": True, "framework": env, "model": str(m.tag)}, option=orjson.OPT_INDENT_2
                    ).decode()
                )
            else:
                click.echo(m.tag)
        return m

    if psutil.WINDOWS:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    return cli


cli = create_cli()

if __name__ == "__main__":
    cli()
