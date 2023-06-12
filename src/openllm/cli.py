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
import re
import subprocess
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
from bentoml._internal.configuration import get_debug_mode, get_quiet_mode, set_quiet_mode
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.log import configure_logging, configure_server_logging
from bentoml_cli.utils import BentoMLCommandGroup

import openllm

from .utils import DEBUG, LazyType, ModelEnv, analytics, bentoml_cattr, first_not_none

if t.TYPE_CHECKING:
    from ._types import ClickFunctionWrapper, F, P

    ServeCommand = t.Literal["serve", "serve-grpc"]
    OutputLiteral = t.Literal["json", "pretty", "porcelain"]

    TupleStrAny = tuple[str, ...]
else:
    TupleStrAny = tuple


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

    llm_config = openllm.AutoConfig.for_model(model_name)

    docstring = f"""\
{llm_config.__openllm_env__.start_docstring}
\b
Available model_id(s): {llm_config.__openllm_model_ids__} [default: {llm_config.__openllm_default_id__}]
"""
    command_attrs: dict[str, t.Any] = {
        "name": llm_config.__openllm_model_name__,
        "context_settings": _context_settings or {},
        "short_help": f"Start a LLMServer for '{model_name}' ('--help' for more details)",
        "help": docstring,
    }

    aliases: list[str] = []
    if llm_config.__openllm_name_type__ == "dasherize":
        aliases.append(llm_config.__openllm_start_name__)

    command_attrs["aliases"] = aliases if len(aliases) > 0 else None

    serve_decorator = _http_server_args if not _serve_grpc else _grpc_server_args

    try:
        llm_config.check_if_gpu_is_available()
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
            analytics.track_start_init(llm_config)
            return llm_config

        return noop

    @group.command(**command_attrs)
    @llm_config.to_click_options
    @serve_decorator
    @cog.optgroup.group("General LLM Options")
    @cog.optgroup.option("--server-timeout", type=int, default=None, help="Server timeout in seconds")
    @model_id_option(cog.optgroup, model_env=llm_config.__openllm_env__)
    @cog.optgroup.option(
        "--device",
        type=tuple,
        cls=NargsOptions,
        nargs=-1,
        envvar="CUDA_VISIBLE_DEVICES",
        callback=parse_device_callback,
        help=f"Assign GPU devices (if available) for {model_name}.",
        show_envvar=True,
    )
    @workers_option(cog.optgroup)
    def model_start(
        server_timeout: int | None,
        model_id: str | None,
        workers: float | None,
        device: tuple[str, ...] | None,
        **attrs: t.Any,
    ) -> openllm.LLMConfig:
        config, server_attrs = llm_config.model_validate_click(**attrs)

        if llm_config.__openllm_env__.get_framework_env() == "flax":
            llm = openllm.AutoFlaxLLM.for_model(model_name, model_id=model_id, llm_config=config)
        elif llm_config.__openllm_env__.get_framework_env() == "tf":
            llm = openllm.AutoTFLLM.for_model(model_name, model_id=model_id, llm_config=config)
        else:
            llm = openllm.AutoLLM.for_model(model_name, model_id=model_id, llm_config=config)

        if llm.config.__openllm_requirements__ is not None and len(llm.config.__openllm_requirements__) > 0:
            _echo(
                f"Make sure to have the following dependencies available: {llm.config.__openllm_requirements__}",
                fg="yellow",
            )

        # NOTE: We need to initialize llm here first to check if the model is already downloaded to
        # avoid deadlock before the subprocess forking.
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "openllm",
                "download-models",
                model_name,
                "--model-id",
                llm.model_id,
                "--output",
                "porcelain",
            ]
        )

        workers_per_resource = first_not_none(workers, default=llm.config.__openllm_workers_per_resource__)
        server_timeout = first_not_none(server_timeout, default=llm.config.__openllm_timeout__)

        num_workers = int(1 / workers_per_resource)
        if num_workers > 1:
            _echo(
                f"{model_name} requires at least {num_workers} GPUs/CPUs available per worker."
                " Make sure that it has available resources to run inference.",
                fg="yellow",
            )

        server_attrs.update({"working_dir": os.path.dirname(__file__)})
        if _serve_grpc:
            server_attrs["grpc_protocol_version"] = "v1"
        # NOTE: currently, theres no development args in bentoml.Server. To be fixed upstream.
        development = server_attrs.pop("development")
        server_attrs.setdefault("production", not development)

        start_env = os.environ.copy()

        # NOTE: This is to set current configuration
        _bentoml_config_options = start_env.pop("BENTOML_CONFIG_OPTIONS", "")
        _bentoml_config_options_opts = [
            "tracing.sample_rate=1.0",
            f"api_server.traffic.timeout={server_timeout}",
            f'runners."llm-{llm.config.__openllm_start_name__}-runner".traffic.timeout={llm.config.__openllm_timeout__}',
            f'runners."llm-{llm.config.__openllm_start_name__}-runner".workers_per_resource={workers_per_resource}',
        ]
        if device:
            if len(device) > 1:
                for idx, dev in enumerate(device):
                    _bentoml_config_options_opts.append(
                        f'runners."llm-{llm.config.__openllm_start_name__}-runner".resources."nvidia.com/gpu"[{idx}]={dev}'
                    )
            else:
                _bentoml_config_options_opts.append(
                    f'runners."llm-{llm.config.__openllm_start_name__}-runner".resources."nvidia.com/gpu"=[{device[0]}]'
                )

        _bentoml_config_options += " " if _bentoml_config_options else "" + " ".join(_bentoml_config_options_opts)

        start_env.update(
            {
                llm.config.__openllm_env__.framework: llm.config.__openllm_env__.get_framework_env(),
                llm.config.__openllm_env__.model_config: llm.config.model_dump_json().decode(),
                "OPENLLM_MODEL": model_name,
                "OPENLLM_MODEL_ID": llm.model_id,
                "BENTOML_DEBUG": str(get_debug_mode()),
                "BENTOML_CONFIG_OPTIONS": _bentoml_config_options,
                "BENTOML_HOME": os.environ.get("BENTOML_HOME", BentoMLContainer.bentoml_home.get()),
            }
        )

        if t.TYPE_CHECKING:
            server_cls: type[bentoml.HTTPServer] if not _serve_grpc else type[bentoml.GrpcServer]

        server_cls = getattr(bentoml, "HTTPServer" if not _serve_grpc else "GrpcServer")
        server_attrs["timeout"] = 90
        server = server_cls("_service.py:svc", **server_attrs)

        try:
            analytics.track_start_init(llm.config)
            server.start(env=start_env, text=True, blocking=True)
        except Exception as err:
            _echo(f"Error caught while starting LLM Server:\n{err}", fg="red")
            raise
        else:
            if not get_debug_mode():
                _echo(
                    f"\n🚀 Next step: run 'openllm build {model_name}' to create a Bento for {model_name}",
                    fg="blue",
                )

        # NOTE: Return the configuration for telemetry purposes.
        return llm_config

    return model_start


class OpenLLMCommandGroup(BentoMLCommandGroup):
    NUMBER_OF_COMMON_PARAMS = 3

    @staticmethod
    def common_params(f: F[P, t.Any]) -> ClickFunctionWrapper[..., t.Any]:
        """This is not supposed to be used with unprocessed click function.
        This should be used a the last currying from common_params -> usage_tracking -> exception_handling
        """
        # The following logics is similar to one of BentoMLCommandGroup

        from bentoml._internal.configuration import DEBUG_ENV_VAR, QUIET_ENV_VAR, set_debug_mode

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
        command_name = attrs.get("name", func.__name__)

        @functools.wraps(func)
        def wrapper(do_not_track: bool, *args: P.args, **attrs: P.kwargs) -> t.Any:
            if do_not_track:
                with analytics.set_bentoml_tracking():
                    return func(*args, **attrs)

            start_time = time.time_ns()

            with analytics.set_bentoml_tracking():
                assert group.name is not None, "group.name should not be None"
                event = analytics.OpenllmCliEvent(cmd_group=group.name, cmd_name=command_name)
                try:
                    return_value = func(*args, **attrs)
                    duration_in_ms = (time.time_ns() - start_time) / 1e6
                    event.duration_in_ms = duration_in_ms
                    analytics.track(event)
                    return return_value
                except Exception as e:
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
        if "max_content_width" not in attrs["context_settings"]:
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
    WrappedServeFunction = ClickFunctionWrapper[t.Concatenate[int, str | None, P], openllm.LLMConfig]
else:
    WrappedServeFunction = t.Any


def parse_serve_args(serve_grpc: bool):
    """Parsing `bentoml serve|serve-grpc` click.Option to be parsed via `openllm start`"""
    from bentoml_cli.cli import cli

    command = "serve" if not serve_grpc else "serve-grpc"
    group = cog.optgroup.group(
        f"Start a {'HTTP' if not serve_grpc else 'gRPC'} server options",
        help=f"Related to serving the model [synonymous to `bentoml {'serve-http' if not serve_grpc else command }`]",
    )

    def decorator(
        f: t.Callable[t.Concatenate[int, str | None, P], openllm.LLMConfig]
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
            f = t.cast("WrappedServeFunction[P]", cog.optgroup.option(*param_decls, **attrs)(f))

        return group(f)

    return decorator


_http_server_args = parse_serve_args(False)
_grpc_server_args = parse_serve_args(True)


class NargsOptions(cog.GroupedOption):
    """An option that supports nargs=-1.
    Derived from https://stackoverflow.com/a/48394004/8643197

    We mk add_to_parser to handle multiple value that is passed into this specific
    options.
    """

    def __init__(self, *args: t.Any, **attrs: t.Any):
        nargs = attrs.pop("nargs", -1)
        if nargs != -1:
            raise openllm.exceptions.OpenLLMException(f"'nargs' is set, and must be -1 instead of {nargs}")
        super(NargsOptions, self).__init__(*args, **attrs)
        self._prev_parser_process: t.Callable[[t.Any, click.parser.ParsingState], None] | None = None
        self._nargs_parser: click.parser.Option | None = None

    def add_to_parser(self, parser: click.OptionParser, ctx: click.Context) -> None:
        def _parser(value: t.Any, state: click.parser.ParsingState):
            # method to hook to the parser.process
            done = False
            value = [value]
            # grab everything up to the next option
            assert self._nargs_parser is not None
            while state.rargs and not done:
                for prefix in self._nargs_parser.prefixes:
                    if state.rargs[0].startswith(prefix):
                        done = True
                if not done:
                    value.append(state.rargs.pop(0))
            value = tuple(value)

            # call the actual process
            assert self._prev_parser_process is not None
            self._prev_parser_process(value, state)

        retval = super(NargsOptions, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._nargs_parser = our_parser
                self._prev_parser_process = our_parser.process
                our_parser.process = _parser
                break
        return retval


def parse_device_callback(_: click.Context, params: click.Parameter, value: tuple[str, ...] | None) -> t.Any:
    if value is None:
        return value

    if not LazyType(TupleStrAny).isinstance(value):
        raise RuntimeError(f"{params} only accept multiple values.")
    parsed: tuple[str, ...] = tuple()
    for v in value:
        if v == ",":
            # NOTE: This hits when CUDA_VISIBLE_DEVICES is set
            continue
        if "," in v:
            parsed += tuple(v.split(","))
        else:
            parsed += tuple(v.split())
    return tuple(filter(lambda x: x, parsed))


def _start(
    model_name: str,
    framework: t.Literal["flax", "tf", "pt"] | None = None,
    **attrs: t.Any,
):
    """Python API to start a LLM server."""
    _serve_grpc = attrs.pop("_serve_grpc", False)

    _ModelEnv = ModelEnv(model_name)

    if framework is not None:
        os.environ[_ModelEnv.framework] = framework
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
    help="Showing output type.",
    show_default=True,
    envvar="OPENLLM_OUTPUT",
    show_envvar=True,
)


def model_id_option(factory: t.Any, model_env: ModelEnv | None = None):
    envvar = None
    if model_env is not None:
        envvar = model_env.model_id
    return factory.option(
        "--model-id",
        type=click.STRING,
        default=None,
        help="Optional model_id name or path for (fine-tune) weight.",
        envvar=envvar,
        show_envvar=True if envvar is not None else False,
    )


def workers_option(factory: t.Any, build: bool = False):
    help_str = """Number of workers per resource assigned.
    See https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy
    for more information. By default, this is set to 1."""
    if build:
        help_str += """\n
    NOTE: The workers value passed into 'build' will determine how the LLM can
    be provisioned in Kubernetes as well as in standalone container. This will
    ensure it has the same effect with 'openllm start --workers ...'"""
    return factory.option(
        "--workers",
        default=None,
        type=click.FLOAT,
        help=help_str,
        required=False,
    )


def cli_factory() -> click.Group:
    from .__about__ import __version__

    configure_logging()

    model_store = BentoMLContainer.model_store.get()

    @click.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="openllm")
    @click.version_option(__version__, "--version", "-v")
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
        An open platform for operating large language models in production.
        Fine-tune, serve, deploy, and monitor any LLMs with ease.
        """

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
    @model_id_option(click)
    @output_option
    @click.option("--overwrite", is_flag=True, help="Overwrite existing Bento for given LLM if it already exists.")
    @workers_option(click, build=True)
    def build(model_name: str, model_id: str | None, overwrite: bool, output: OutputLiteral, workers: float | None):
        """Package a given models into a Bento.

        $ openllm build flan-t5

        \b
        NOTE: To run a container built from this Bento with GPU support, make sure
        to have https://github.com/NVIDIA/nvidia-container-toolkit install locally.
        """
        if output == "porcelain":
            set_quiet_mode(True)
            configure_server_logging()

        bento, _previously_built = openllm.build(
            model_name,
            __cli__=True,
            model_id=model_id,
            _workers=workers,
            _overwrite_existing_bento=overwrite,
        )

        if output == "pretty":
            if not get_quiet_mode():
                _echo("\n" + OPENLLM_FIGLET, fg="white")
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
                    + f"    $ bentoml containerize {bento.tag}\n"
                    + "    Tip: To enable additional BentoML feature for 'containerize', "
                    + "use '--enable-features=FEATURE[,FEATURE]' "
                    + "[see 'bentoml containerize -h' for more advanced usage]\n",
                    fg="blue",
                )
        elif output == "json":
            _echo(orjson.dumps(bento.info.to_dict(), option=orjson.OPT_INDENT_2).decode())
        else:
            _echo(bento.tag)
        return bento

    @cli.command()
    @output_option
    @click.option(
        "--show-available",
        is_flag=True,
        default=False,
        help="Show available models in local store (mutually exclusive with '-o porcelain').",
    )
    def models(output: OutputLiteral, show_available: bool):
        """List all supported models.

        NOTE: '--show-available' and '-o porcelain' are mutually exclusive.
        """
        from ._llm import convert_transformers_model_name

        models = tuple(inflection.dasherize(key) for key in openllm.CONFIG_MAPPING.keys())
        if output == "porcelain":
            if show_available:
                raise click.BadOptionUsage(
                    "--show-available", "Cannot use '--show-available' with '-o porcelain' (mutually exclusive)."
                )
            _echo("\n".join(models), fg="white")
        else:
            failed_initialized: list[tuple[str, Exception]] = []

            json_data: dict[
                str, dict[t.Literal["model_id", "url", "installation", "requires_gpu", "runtime_impl"], t.Any]
            ] = {}

            # NOTE: Keep a sync list with ./tools/update-optional-dependencies.py
            extras = ["chatglm", "falcon", "flan-t5", "starcoder"]

            converted: list[str] = []
            for m in models:
                config = openllm.AutoConfig.for_model(m)
                runtime_impl: tuple[t.Literal["pt", "flax", "tf"], ...] = tuple()
                if config.__openllm_model_name__ in openllm.MODEL_MAPPING_NAMES:
                    runtime_impl += ("pt",)
                if config.__openllm_model_name__ in openllm.MODEL_FLAX_MAPPING_NAMES:
                    runtime_impl += ("flax",)
                if config.__openllm_model_name__ in openllm.MODEL_TF_MAPPING_NAMES:
                    runtime_impl += ("tf",)
                json_data[m] = {
                    "model_id": config.__openllm_model_ids__,
                    "url": config.__openllm_url__,
                    "requires_gpu": config.__openllm_requires_gpu__,
                    "runtime_impl": runtime_impl,
                    "installation": "pip install openllm" if m not in extras else f'pip install "openllm[{m}]"',
                }
                converted.extend([convert_transformers_model_name(i) for i in config.__openllm_model_ids__])
                if DEBUG:
                    try:
                        openllm.AutoLLM.for_model(m, llm_config=config)
                    except Exception as err:
                        failed_initialized.append((m, err))

            ids_in_local_store = None
            if show_available:
                ids_in_local_store = [i for i in bentoml.models.list() if any(n in i.tag.name for n in converted)]

            if output == "pretty":
                import tabulate

                tabulate.PRESERVE_WHITESPACE = True

                data: list[
                    str | tuple[str, str, list[str], str, t.LiteralString, tuple[t.Literal["pt", "flax", "tf"], ...]]
                ] = []
                for m, v in json_data.items():
                    data.extend(
                        [
                            (
                                m,
                                v["url"],
                                v["model_id"],
                                v["installation"],
                                "✅" if v["requires_gpu"] else "❌",
                                v["runtime_impl"],
                            )
                        ]
                    )
                column_widths = [
                    int(COLUMNS / 6),
                    int(COLUMNS / 6),
                    int(COLUMNS / 3),
                    int(COLUMNS / 6),
                    int(COLUMNS / 6),
                    int(COLUMNS / 9),
                ]

                if len(data) == 0 and len(failed_initialized) > 0:
                    _echo("Exception found while parsing models:\n", fg="yellow")
                    for m, err in failed_initialized:
                        _echo(f"- {m}: ", fg="yellow", nl=False)
                        _echo(traceback.print_exception(err, limit=3), fg="red")
                    sys.exit(1)

                table = tabulate.tabulate(
                    data,
                    tablefmt="fancy_grid",
                    headers=["LLM", "URL", "Models Id", "Installation", "GPU Only", "Runtime"],
                    maxcolwidths=column_widths,
                )

                formatted_table = ""
                for line in table.split("\n"):
                    formatted_table += (
                        "".join(f"{cell:{width}}" for cell, width in zip(line.split("\t"), column_widths)) + "\n"
                    )
                _echo(formatted_table, fg="white")

                if DEBUG and len(failed_initialized) > 0:
                    _echo("\nThe following models are supported but failed to initialize:\n")
                    for m, err in failed_initialized:
                        _echo(f"- {m}: ", fg="blue", nl=False)
                        _echo(err, fg="red")

                if show_available:
                    assert ids_in_local_store
                    _echo("The following models are available in local store:\n", fg="white")
                    for i in ids_in_local_store:
                        _echo(f"- {i}", fg="white")
            else:
                dumped: dict[str, t.Any] = json_data
                if show_available:
                    assert ids_in_local_store
                    dumped["local"] = [bentoml_cattr.unstructure(i.tag) for i in ids_in_local_store]
                _echo(
                    orjson.dumps(
                        dumped,
                        option=orjson.OPT_INDENT_2,
                    ).decode(),
                    fg="white",
                )

        sys.exit(0)

    @cli.command()
    @click.argument(
        "model_name", type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()])
    )
    @model_id_option(click)
    @output_option
    def download_models(model_name: str, model_id: str | None, output: OutputLiteral):
        """Setup LLM interactively.

        Note: This is useful for development and setup for fine-tune.
        """
        config = openllm.AutoConfig.for_model(model_name)
        env = config.__openllm_env__.get_framework_env()
        if env == "flax":
            model = openllm.AutoFlaxLLM.for_model(model_name, model_id=model_id, llm_config=config)
        elif env == "tf":
            model = openllm.AutoTFLLM.for_model(model_name, model_id=model_id, llm_config=config)
        else:
            model = openllm.AutoLLM.for_model(model_name, model_id=model_id, llm_config=config)

        tag = model.make_tag(trust_remote_code=config.__openllm_trust_remote_code__)

        if len(bentoml.models.list(tag)) == 0:
            if output == "pretty":
                _echo(f"{tag} does not exists yet!. Downloading...", fg="yellow", nl=True)
            m = model.ensure_model_id_exists()
            if output == "pretty":
                _echo(f"Saved model: {m.tag}")
            elif output == "json":
                _echo(
                    orjson.dumps(
                        {"previously_setup": False, "framework": env, "tag": str(m.tag)}, option=orjson.OPT_INDENT_2
                    ).decode()
                )
            else:
                _echo(tag)
        else:
            m = bentoml.transformers.get(tag)
            if output == "pretty":
                _echo(f"{model_name} is already setup for framework '{env}': {str(m.tag)}", nl=True, fg="yellow")
            elif output == "json":
                _echo(
                    orjson.dumps(
                        {"previously_setup": True, "framework": env, "model": str(m.tag)}, option=orjson.OPT_INDENT_2
                    ).decode(),
                    fg="white",
                )
            else:
                _echo(m.tag, fg="white")
        return m

    @cli.command()
    @click.option(
        "-y",
        "--yes",
        "--assume-yes",
        is_flag=True,
        help="Skip confirmation when deleting a specific model",
    )
    def prune(yes: bool):
        """Remove all saved models locally."""
        available = [
            m
            for t in map(inflection.dasherize, openllm.CONFIG_MAPPING.keys())
            for m in bentoml.models.list()
            if t in m.tag.name
        ]

        for model in available:
            if yes:
                delete_confirmed = True
            else:
                delete_confirmed = click.confirm(f"delete model {model.tag}?")

            if delete_confirmed:
                model_store.delete(model.tag)
                click.echo(f"{model} deleted.")

    @cli.command(name="query")
    @click.option(
        "--endpoint",
        type=click.STRING,
        help="OpenLLM Server endpoint, i.e: http://0.0.0.0:3000",
        envvar="OPENLLM_ENDPOINT",
        default="http://0.0.0.0:3000",
    )
    @click.option("--timeout", type=click.INT, default=30, help="Default server timeout", show_default=True)
    @click.option(
        "--server-type", type=click.Choice(["grpc", "http"]), help="Server type", default="http", show_default=True
    )
    @output_option
    @click.argument("query", type=click.STRING)
    def query_(
        query: str,
        endpoint: str,
        timeout: int,
        server_type: t.Literal["http", "grpc"],
        output: OutputLiteral,
    ):
        """Ask a LLM interactively, from a terminal.

        $ openllm query --endpoint http://12.323.2.1:3000 "What is the meaning of life?"
        """
        if server_type == "grpc":
            endpoint = re.sub(r"http://", "", endpoint)
        client = (
            openllm.client.HTTPClient(endpoint, timeout=timeout)
            if server_type == "http"
            else openllm.client.GrpcClient(endpoint, timeout=timeout)
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
        assert download_models and build and models and start and start_grpc and query_ and prune

    if psutil.WINDOWS:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore

    return cli


cli = cli_factory()

if __name__ == "__main__":
    cli()
