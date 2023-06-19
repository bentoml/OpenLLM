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

This extends BentoML's internal CLI CommandGroup.
"""
from __future__ import annotations

import functools
import inspect
import itertools
import logging
import os
import re
import sys
import time
import traceback
import typing as t

import click
import click_option_group as cog
import inflection
import orjson
import psutil
from bentoml_cli.utils import BentoMLCommandGroup
from bentoml_cli.utils import opt_callback
from simple_di import Provide
from simple_di import inject

import bentoml
import openllm
from bentoml._internal.configuration.containers import BentoMLContainer

from .__about__ import __version__
from .exceptions import OpenLLMException
from .utils import DEBUG
from .utils import LazyLoader
from .utils import LazyType
from .utils import ModelEnv
from .utils import analytics
from .utils import bentoml_cattr
from .utils import configure_logging
from .utils import first_not_none
from .utils import get_debug_mode
from .utils import get_quiet_mode
from .utils import gpu_count
from .utils import is_torch_available
from .utils import is_transformers_supports_agent
from .utils import set_debug_mode
from .utils import set_quiet_mode


if t.TYPE_CHECKING:
    import torch

    from bentoml._internal.models import ModelStore

    from ._types import ClickFunctionWrapper
    from ._types import F
    from ._types import P
    from .models.auto.factory import _BaseAutoLLMClass

    ServeCommand = t.Literal["serve", "serve-grpc"]
    OutputLiteral = t.Literal["json", "pretty", "porcelain"]

    TupleStrAny = tuple[str, ...]
else:
    TupleStrAny = tuple
    torch = LazyLoader("torch", globals(), "torch")


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


class NargsOptions(cog.GroupedOption):
    """An option that supports nargs=-1.
    Derived from https://stackoverflow.com/a/48394004/8643197

    We mk add_to_parser to handle multiple value that is passed into this specific
    options.
    """

    def __init__(self, *args: t.Any, **attrs: t.Any):
        nargs = attrs.pop("nargs", -1)
        if nargs != -1:
            raise OpenLLMException(f"'nargs' is set, and must be -1 instead of {nargs}")
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


def parse_device_callback(
    _: click.Context, params: click.Parameter, value: tuple[str, ...] | tuple[t.Literal["all"] | str] | None
) -> t.Any:
    if value is None:
        return value

    if not LazyType(TupleStrAny).isinstance(value):
        raise RuntimeError(f"{params} only accept multiple values.")

    # NOTE: --device all is a special case
    if len(value) == 1 and value[0] == "all":
        return gpu_count()

    parsed: tuple[str, ...] = ()
    for v in value:
        if v == ",":
            # NOTE: This hits when CUDA_VISIBLE_DEVICES is set
            continue
        if "," in v:
            parsed += tuple(v.split(","))
        else:
            parsed += tuple(v.split())
    return tuple(filter(lambda x: x, parsed))


def _echo(text: t.Any, fg: str = "green", _with_style: bool = True, **attrs: t.Any) -> None:
    call = click.echo
    if _with_style:
        attrs["fg"] = fg if not get_debug_mode() else None
        call = click.secho
    call(text, **attrs)


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


def model_id_option(factory: t.Any, model_env: ModelEnv | None = None, click_type: click.ParamType | None = None):
    envvar = None
    if model_env is not None:
        envvar = model_env.model_id
    return factory.option(
        "--model-id",
        type=click_type if click_type else click.STRING,
        default=None,
        help="Optional model_id name or path for (fine-tune) weight.",
        envvar=envvar,
        show_envvar=True if envvar is not None else False,
    )


def workers_per_resource_option(factory: t.Any, build: bool = False):
    help_str = """Number of workers per resource assigned.
    See https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy
    for more information. By default, this is set to 1."""
    if build:
        help_str += """\n
    NOTE: The workers value passed into 'build' will determine how the LLM can
    be provisioned in Kubernetes as well as in standalone container. This will
    ensure it has the same effect with 'openllm start --workers ...'"""
    return factory.option(
        "--workers-per-resource",
        default=None,
        type=click.FLOAT,
        help=help_str,
        required=False,
    )


def quantize_option(factory: t.Any, build: bool = False):
    help_str = (
        "Running this model in quantized mode." if not build else "Set quantization mode for serving in deployment."
    )
    help_str += """\n

    GPTQ is currently working in progress and will be available soon.

    NOTE: Quantization is only available for PyTorch models.
    """
    return factory.option(
        "--quantize",
        type=click.Choice(["int8", "int4", "gptq"]),
        default=None,
        help=help_str,
    )


def bettertransformer_option(factory: t.Any, model_env: ModelEnv | None = None):
    envvar = None
    if model_env is not None:
        envvar = model_env.model_id
    return factory.option(
        "--bettertransformer",
        is_flag=True,
        default=None,
        help="Use BetterTransformer wrapper to serve model. This will applies during serving time.",
        envvar=envvar,
        show_envvar=True if envvar is not None else False,
    )


class OpenLLMCommandGroup(BentoMLCommandGroup):
    NUMBER_OF_COMMON_PARAMS = 3

    @staticmethod
    def common_params(f: F[P, t.Any]) -> ClickFunctionWrapper[..., t.Any]:
        """This is not supposed to be used with unprocessed click function.
        This should be used a the last currying from common_params -> usage_tracking -> exception_handling
        """
        # The following logics is similar to one of BentoMLCommandGroup

        from bentoml._internal.configuration import DEBUG_ENV_VAR
        from bentoml._internal.configuration import QUIET_ENV_VAR

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
            except OpenLLMException as err:
                raise click.ClickException(
                    click.style(f"[{group.name}] '{command_name}' failed: " + err.message, fg="red")
                ) from err
            except KeyboardInterrupt:  # NOTE: silience KeyboardInterrupt
                pass

        return t.cast("ClickFunctionWrapper[..., t.Any]", wrapper)

    def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
        cmd_name = self.resolve_alias(cmd_name)
        if ctx.command.name == "start":
            try:
                return _cached_http[cmd_name]
            except KeyError:
                raise click.BadArgumentUsage(f"{cmd_name} is not a valid model identifier supported by OpenLLM.")
        elif ctx.command.name == "start-grpc":
            try:
                return _cached_grpc[cmd_name]
            except KeyError:
                raise click.BadArgumentUsage(f"{cmd_name} is not a valid model identifier supported by OpenLLM.")
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
            if "do_not_track" in attrs:
                # We hit this branch when ctx.invoke the function
                attrs.pop("do_not_track")
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


@cli.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="start")
def start_cli():
    """
    Start any LLM as a REST server.

    \b
    ```bash
    $ openllm start <model_name> --<options> ...
    ```
    """


@cli.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="start-grpc")
def start_grpc_cli():
    """
    Start any LLM as a gRPC server.

    \b
    ```bash
    $ openllm start-grpc <model_name> --<options> ...
    ```
    """


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


def start_model_command(
    model_name: str,
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

    configure_logging()

    llm_config = openllm.AutoConfig.for_model(model_name)
    env: ModelEnv = llm_config["env"]

    docstring = f"""\
{env.start_docstring}
\b
Available model_id(s): {llm_config['model_ids']} [default: {llm_config['default_id']}]
"""
    command_attrs: dict[str, t.Any] = {
        "name": llm_config["model_name"],
        "context_settings": _context_settings or {},
        "short_help": f"Start a LLMServer for '{model_name}' ('--help' for more details)",
        "help": docstring,
    }

    aliases: list[str] = []
    if llm_config["name_type"] == "dasherize":
        aliases.append(llm_config["start_name"])

    command_attrs["aliases"] = aliases if len(aliases) > 0 else None

    serve_decorator = _http_server_args if not _serve_grpc else _grpc_server_args
    group = start_cli if not _serve_grpc else start_grpc_cli

    available_gpu = gpu_count()
    if llm_config["requires_gpu"] and len(available_gpu) < 1:
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
    @cog.optgroup.option(
        "--server-timeout",
        type=int,
        default=None,
        help="Server timeout in seconds",
    )
    @workers_per_resource_option(cog.optgroup)
    @model_id_option(cog.optgroup, model_env=env, click_type=click.Choice(llm_config["model_ids"]))
    @cog.optgroup.option(
        "--fast",
        is_flag=True,
        default=False,
        help="Bypass auto model checks and setup. This option is ahead-of-serving time.",
    )
    @cog.optgroup.group("LLM Optimization Options.")
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
    @quantize_option(cog.optgroup)
    @bettertransformer_option(cog.optgroup, model_env=env)
    @click.pass_context
    def model_start(
        ctx: click.Context,
        server_timeout: int | None,
        model_id: str | None,
        workers_per_resource: float | None,
        device: tuple[str, ...] | None,
        quantize: t.Literal["int8", "int4", "gptq"] | None,
        bettertransformer: bool | None,
        fast: bool,
        **attrs: t.Any,
    ) -> openllm.LLMConfig:
        config, server_attrs = llm_config.model_validate_click(**attrs)

        # Create a new model env to work with the envvar during CLI invocation
        env = ModelEnv(config["model_name"])
        framework_envvar = env.framework_value

        if quantize:
            gpu_available = gpu_count()
            if len(gpu_available) < 1:
                _echo(f"Quantization requires at least 1 GPU (got {len(gpu_available)})", fg="red")
                ctx.exit(1)
            if framework_envvar != "pt":
                _echo("Quantization is currently only available for PyTorch models.", fg="red")
                ctx.exit(1)

        # We need to handle None separately here, as env from subprocess doesn't
        # accept None value.
        env = ModelEnv(env.model_name, bettertransformer=bettertransformer, quantize=quantize)

        requirements = config["requirements"]
        if requirements is not None and len(requirements) > 0:
            _echo(
                f"Make sure to have the following dependencies available: {requirements}",
                fg="yellow",
            )

        workers_per_resource = first_not_none(workers_per_resource, default=config["workers_per_resource"])
        server_timeout = first_not_none(server_timeout, default=config["timeout"])

        num_workers = int(1 / workers_per_resource)
        if num_workers > 1:
            _echo(
                f"Running '{model_name}' requires at least {num_workers} GPUs/CPUs available per worker."
                " Make sure that it has available resources for inference.",
                fg="yellow",
            )

        server_attrs.update({"working_dir": os.path.dirname(__file__)})
        if _serve_grpc:
            server_attrs["grpc_protocol_version"] = "v1"
        # NOTE: currently, theres no development args in bentoml.Server. To be fixed upstream.
        development = server_attrs.pop("development")
        server_attrs.setdefault("production", not development)

        # NOTE: This is to set current configuration
        start_env = os.environ.copy()
        _bentoml_config_options_env = start_env.pop("BENTOML_CONFIG_OPTIONS", "")
        _bentoml_config_options_opts = [
            "tracing.sample_rate=1.0",
            f"api_server.traffic.timeout={server_timeout}",
            f'runners."llm-{config["start_name"]}-runner".traffic.timeout={config["timeout"]}',
            f'runners."llm-{config["start_name"]}-runner".workers_per_resource={workers_per_resource}',
        ]
        if device:
            if len(device) > 1:
                for idx, dev in enumerate(device):
                    _bentoml_config_options_opts.append(
                        f'runners."llm-{config["start_name"]}-runner".resources."nvidia.com/gpu"[{idx}]={dev}'
                    )
            else:
                _bentoml_config_options_opts.append(
                    f'runners."llm-{config["start_name"]}-runner".resources."nvidia.com/gpu"=[{device[0]}]'
                )

        _bentoml_config_options_env += (
            " " if _bentoml_config_options_env else "" + " ".join(_bentoml_config_options_opts)
        )

        if fast and not get_quiet_mode():
            _echo(
                f"Make sure to download the model before 'start': 'openllm download {model_name}{'--model-id ' + model_id if model_id else ''}'",
                fg="yellow",
            )
        automodel_attrs = {
            "model_id": model_id,
            "llm_config": config,
            "ensure_available": not fast,
        }

        if framework_envvar == "pt":
            automodel_attrs.update({"quantize": quantize, "bettertransformer": bettertransformer})

        llm = t.cast(
            "_BaseAutoLLMClass",
            openllm[framework_envvar],  # type: ignore (internal API)
        ).for_model(model_name, **automodel_attrs)

        start_env.update(
            {
                env.framework: env.framework_value,
                env.config: llm.config.model_dump_json().decode(),
                "OPENLLM_MODEL": model_name,
                "OPENLLM_MODEL_ID": llm.model_id,
                "BENTOML_DEBUG": str(get_debug_mode()),
                "BENTOML_CONFIG_OPTIONS": _bentoml_config_options_env,
                "BENTOML_HOME": os.environ.get("BENTOML_HOME", BentoMLContainer.bentoml_home.get()),
            }
        )

        if env.bettertransformer_value is not None:
            start_env[env.bettertransformer] = env.bettertransformer_value
        if env.quantize_value is not None:
            start_env[env.quantize] = env.quantize_value

        if t.TYPE_CHECKING:
            server_cls: type[bentoml.HTTPServer] if not _serve_grpc else type[bentoml.GrpcServer]

        server_cls = getattr(bentoml, "HTTPServer" if not _serve_grpc else "GrpcServer")
        server_attrs["timeout"] = server_timeout
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


_cached_http = {key: start_model_command(key, _context_settings=_CONTEXT_SETTINGS) for key in openllm.CONFIG_MAPPING}
_cached_grpc = {
    key: start_model_command(key, _context_settings=_CONTEXT_SETTINGS, _serve_grpc=True)
    for key in openllm.CONFIG_MAPPING
}


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
    start_model_command(model_name, _serve_grpc=_serve_grpc)(standalone_mode=False, **attrs)


start = functools.partial(_start, _serve_grpc=False)
start_grpc = functools.partial(_start, _serve_grpc=True)


@cli.command()
@click.argument(
    "model_name", type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()])
)
@model_id_option(click)
@output_option
@click.option("--overwrite", is_flag=True, help="Overwrite existing Bento for given LLM if it already exists.")
@workers_per_resource_option(click, build=True)
@cog.optgroup.group(cls=cog.MutuallyExclusiveOptionGroup, name="Optimisation options.")
@quantize_option(cog.optgroup, build=True)
@bettertransformer_option(cog.optgroup)
@click.option(
    "--enable-features",
    help="Enable additional features for building this LLM Bento. Available: {}".format(
        ", ".join(openllm.utils.OPTIONAL_DEPENDENCIES)
    ),
    multiple=True,
    nargs=1,
    metavar="FEATURE[,FEATURE]",
)
def build(
    model_name: str,
    model_id: str | None,
    overwrite: bool,
    output: OutputLiteral,
    quantize: t.Literal["int8", "int4", "gptq"] | None,
    enable_features: tuple[str] | None,
    bettertransformer: bool | None,
    workers_per_resource: float | None,
):
    """Package a given models into a Bento.

    \b
    ```bash
    $ openllm build flan-t5 --model-id google/flan-t5-large
    ```

    \b
    > NOTE: To run a container built from this Bento with GPU support, make sure
    > to have https://github.com/NVIDIA/nvidia-container-toolkit install locally.
    """
    if output == "porcelain":
        set_quiet_mode(True)
        configure_logging()

    if output == "pretty":
        if overwrite:
            _echo(f"Overwriting existing Bento for {model_name}.", fg="yellow")

    if enable_features:
        enable_features = tuple(itertools.chain.from_iterable((s.split(",") for s in enable_features)))

    bento, _previously_built = openllm.build(
        model_name,
        __cli__=True,
        model_id=model_id,
        quantize=quantize,
        bettertransformer=bettertransformer,
        _extra_dependencies=enable_features,
        _workers_per_resource=workers_per_resource,
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

    \b
    > NOTE: '--show-available' and '-o porcelain' are mutually exclusive.

    \b
    ```bash
    openllm models --show-available
    ```
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
            runtime_impl: tuple[t.Literal["pt", "flax", "tf"] | str, ...] = ()
            if config["model_name"] in openllm.MODEL_MAPPING_NAMES:
                runtime_impl += ("pt",)
            if config["model_name"] in openllm.MODEL_FLAX_MAPPING_NAMES:
                runtime_impl += ("flax",)
            if config["model_name"] in openllm.MODEL_TF_MAPPING_NAMES:
                runtime_impl += ("tf",)
            json_data[m] = {
                "model_id": config["model_ids"],
                "url": config["url"],
                "requires_gpu": config["requires_gpu"],
                "runtime_impl": runtime_impl,
                "installation": "pip install openllm" if m not in extras else f'pip install "openllm[{m}]"',
            }
            converted.extend([convert_transformers_model_name(i) for i in config["model_ids"]])
            if DEBUG:
                try:
                    openllm.AutoLLM.for_model(m, llm_config=config)
                except Exception as err:
                    failed_initialized.append((m, err))

        ids_in_local_store = None
        if show_available:
            ids_in_local_store = {k: [i for i in bentoml.models.list() if k in i.tag.name] for k in json_data.keys()}
            ids_in_local_store = {k: v for k, v in ids_in_local_store.items() if v}

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
                int(COLUMNS / 3),
                int(COLUMNS / 4),
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

                _available = [[k + "\n\n" * len(v), [str(i.tag) for i in v]] for k, v in ids_in_local_store.items()]
                column_widths = [int(COLUMNS / 6), int(COLUMNS / 2)]
                table = tabulate.tabulate(
                    _available,
                    tablefmt="fancy_grid",
                    headers=["Model Id", "Models"],
                    maxcolwidths=column_widths,
                )
                _echo("The following models are available in local store:\n", fg="magenta")

                formatted_table = ""
                for line in table.split("\n"):
                    formatted_table += (
                        "".join(f"{cell:{width}}" for cell, width in zip(line.split("\t"), column_widths)) + "\n"
                    )
                _echo(formatted_table, fg="white")
        else:
            dumped: dict[str, t.Any] = json_data
            if show_available:
                assert ids_in_local_store
                dumped["local"] = [bentoml_cattr.unstructure(i.tag) for m in ids_in_local_store.values() for i in m]
            _echo(
                orjson.dumps(
                    dumped,
                    option=orjson.OPT_INDENT_2,
                ).decode(),
                fg="white",
            )

    sys.exit(0)


@cli.command()
@click.option(
    "-y",
    "--yes",
    "--assume-yes",
    is_flag=True,
    help="Skip confirmation when deleting a specific model",
)
@inject
def prune(yes: bool, model_store: ModelStore = Provide[BentoMLContainer.model_store]):
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


# Can also support agent type such as langchain and other variants here.
_AGENT_MAPPING = {"hf": "/hf/agent"}


def parsing_instruction_callback(
    ctx: click.Context, param: click.Parameter, value: list[str] | str | None
) -> tuple[str, bool | str] | list[str] | str | None:
    if value is None:
        return value

    if isinstance(value, list):
        # we only parse --text foo bar -> --text foo and omit bar
        value = value[-1]
    if not isinstance(value, str):
        raise click.BadParameter(f"Invalid option format: {value}")

    key, *values = value.split("=")
    if not key.startswith("--"):
        raise click.BadParameter(f"Invalid option format: {value}")
    key = key[2:]
    if len(values) == 0:
        return key, True
    elif len(values) == 1:
        return key, values[0]
    else:
        raise click.BadParameter(f"Invalid option format: {value}")


def shared_client_options(f: t.Callable[..., t.Any]) -> t.Callable[..., t.Any]:
    options = [
        click.option(
            "--endpoint",
            type=click.STRING,
            help="OpenLLM Server endpoint, i.e: http://localhost:3000",
            envvar="OPENLLM_ENDPOINT",
            default="http://localhost:3000",
        ),
        click.option("--timeout", type=click.INT, default=30, help="Default server timeout", show_default=True),
        output_option,
    ]
    for opt in reversed(options):
        f = opt(f)
    return f


@cli.command()
@click.argument("task", type=click.STRING, metavar="TASK")
@shared_client_options
@click.option(
    "--agent",
    type=click.Choice(["hf"]),
    default="hf",
    help="Whether to interact with Agents from given Server endpoint.",
    show_default=True,
)
@click.option(
    "--remote",
    is_flag=True,
    default=False,
    help="Whether or not to use remote tools (inference endpoints) instead of local ones.",
    show_default=True,
)
@click.option(
    "--opt",
    help="Define prompt options. "
    "(format: ``--opt text='I love this' --opt audio:./path/to/audio  --opt image:/path/to/file``)",
    required=False,
    multiple=True,
    callback=opt_callback,
    metavar="ARG=VALUE[,ARG=VALUE]",
)
def instruct(
    endpoint: str,
    timeout: int,
    agent: t.LiteralString,
    output: OutputLiteral,
    remote: bool,
    task: str,
    _memoized: dict[str, t.Any],
    **attrs: t.Any,
):
    """Instruct agents interactively for given tasks, from a terminal

    \b
    ```bash
    $ openllm instruct --endpoint http://12.323.2.1:3000 \\
        "Is the following `text` (in Spanish) positive or negative?" \\
        --text "¡Este es un API muy agradable!"
    ```
    """
    client = openllm.client.HTTPClient(endpoint, timeout=timeout)

    if agent == "hf":
        if not is_transformers_supports_agent():
            raise click.UsageError(
                "Transformers version should be at least 4.29 to support HfAgent. "
                "Upgrade with 'pip install -U transformers'"
            )

        _memoized = {k: v[0] for k, v in _memoized.items() if v}

        client._hf_agent.set_stream(logger.info)
        if output != "porcelain":
            _echo(f"Sending the following prompt ('{task}') with the following vars: {_memoized}", fg="magenta")

        result = client.ask_agent(task, agent_type=agent, return_code=False, remote=remote, **_memoized)
        if output == "json":
            _echo(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode(), fg="white")
        else:
            _echo(result, fg="white")
        return result
    else:
        raise click.BadOptionUsage("agent", f"Unknown agent type {agent}")


@cli.command()
@shared_client_options
@click.option(
    "--server-type", type=click.Choice(["grpc", "http"]), help="Server type", default="http", show_default=True
)
@click.argument("prompt", type=click.STRING)
def query(
    prompt: str,
    endpoint: str,
    timeout: int,
    server_type: t.Literal["http", "grpc"],
    output: OutputLiteral,
):
    """Ask a LLM interactively, from a terminal.

    \b
    ```bash
    $ openllm query --endpoint http://12.323.2.1:3000 "What is the meaning of life?"
    ```
    """
    if server_type == "grpc":
        endpoint = re.sub(r"http://", "", endpoint)
    client = (
        openllm.client.HTTPClient(endpoint, timeout=timeout)
        if server_type == "http"
        else openllm.client.GrpcClient(endpoint, timeout=timeout)
    )

    model = t.cast(
        "_BaseAutoLLMClass",
        openllm[client.framework],  # type: ignore (internal API)
    ).for_model(client.model_name)

    if output != "porcelain":
        _echo(f"Processing query: {prompt}", fg="white")

    res = client.query(prompt, return_raw_response=True)

    if output == "pretty":
        formatted = model.postprocess_generate(prompt, res["responses"])
        _echo("Responses: ", fg="white", nl=False)
        _echo(formatted, fg="cyan")
    elif output == "json":
        _echo(orjson.dumps(res, option=orjson.OPT_INDENT_2).decode(), fg="white")
    else:
        _echo(res["responses"], fg="white")


@cli.command(name="download")
@click.argument(
    "model_name",
    type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()]),
)
@model_id_option(click)
@output_option
def download_models(model_name: str, model_id: str | None, output: OutputLiteral):
    """Setup LLM interactively.

    \b
    Note: This is useful for development and setup for fine-tune.

    \b
    ```bash
    $ openllm download opt --model-id facebook/opt-2.7b
    ```
    """
    if output == "porcelain":
        set_quiet_mode(True)
        configure_logging()

    config = openllm.AutoConfig.for_model(model_name)
    envvar = config["env"]["framework_value"]
    model = t.cast(
        "_BaseAutoLLMClass",
        openllm[envvar],  # type: ignore (internal API)
    ).for_model(model_name, model_id=model_id, llm_config=config)

    try:
        _ref = bentoml.transformers.get(model.tag)
        if output == "pretty":
            _echo(f"{model_name} is already setup for framework '{envvar}': {str(_ref.tag)}", nl=True, fg="yellow")
        elif output == "json":
            _echo(
                orjson.dumps(
                    {"previously_setup": True, "framework": envvar, "model": str(_ref.tag)}, option=orjson.OPT_INDENT_2
                ).decode(),
                fg="white",
            )
        else:
            if DEBUG or get_debug_mode():
                # NOTE: When debug is enabled,
                # We will prefix the tag with __tag__ and we can use regex to correctly
                # get the tag from 'bentoml.bentos.build|build_bentofile'
                _echo(f"__tag__:{_ref.tag}", fg="white")
            else:
                _echo(_ref.tag, fg="white")
    except bentoml.exceptions.NotFound:
        if output == "pretty":
            _echo(
                f"'{model.__class__.__name__}' with tag '{model.tag}'"
                " does not exists in local store!. Saving to store...",
                fg="yellow",
                nl=True,
            )

        (model_args, model_attrs), tokenizer_attrs = model.llm_parameters
        _ref = model.import_model(
            model.model_id,
            model.tag,
            *model_args,
            tokenizer_kwds=tokenizer_attrs,
            trust_remote_code=model.__llm_trust_remote_code__,
            **model_attrs,
        )
        if output == "pretty":
            _echo(f"Saved model: {_ref.tag}")
        elif output == "json":
            _echo(
                orjson.dumps(
                    {"previously_setup": False, "framework": envvar, "tag": str(_ref.tag)},
                    option=orjson.OPT_INDENT_2,
                ).decode()
            )
        else:
            if DEBUG or get_debug_mode():
                # NOTE: When debug is enabled,
                # We will prefix the tag with __tag__ and we can use regex to correctly
                # get the tag from 'bentoml.bentos.build|build_bentofile'
                _echo(f"__tag__:{_ref.tag}")
            else:
                _echo(_ref.tag)
    finally:
        if is_torch_available() and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return _ref


if psutil.WINDOWS:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


if __name__ == "__main__":
    cli()
