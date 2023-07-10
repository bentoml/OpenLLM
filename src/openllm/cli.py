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
"""CLI utilities for OpenLLM.

This module also contains the SDK to call ``start`` and ``build`` from SDK

Start any LLM:

```python
openllm.start("falcon", model_id='tiiuae/falcon-7b-instruct')
```

Build a BentoLLM

```python
bento = openllm.build("falcon")
```

Import any LLM into local store
```python
bentomodel = openllm.import_model("falcon", model_id='tiiuae/falcon-7b-instruct')
```
"""
from __future__ import annotations
import functools
import importlib.util
import inspect
import itertools
import logging
import os
import pkgutil
import re
import subprocess
import sys
import tempfile
import time
import traceback
import typing as t

import click
import click_option_group as cog
import fs
import fs.errors
import inflection
import orjson
import psutil
import yaml
from bentoml_cli.utils import BentoMLCommandGroup
from bentoml_cli.utils import opt_callback
from simple_di import Provide
from simple_di import inject

import bentoml
import openllm
from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.model import ModelStore

from .__about__ import __version__
from .exceptions import OpenLLMException
from .utils import DEBUG
from .utils import EnvVarMixin
from .utils import LazyLoader
from .utils import LazyType
from .utils import analytics
from .utils import bentoml_cattr
from .utils import codegen
from .utils import configure_logging
from .utils import dantic
from .utils import first_not_none
from .utils import get_debug_mode
from .utils import get_quiet_mode
from .utils import gpu_count
from .utils import is_jupyter_available
from .utils import is_jupytext_available
from .utils import is_notebook_available
from .utils import is_peft_available
from .utils import is_torch_available
from .utils import is_transformers_supports_agent
from .utils import resolve_user_filepath
from .utils import set_debug_mode
from .utils import set_quiet_mode


if t.TYPE_CHECKING:
    import torch

    from ._types import AnyCallable
    from ._types import ClickFunctionWrapper
    from ._types import DictStrAny
    from ._types import ListStr
    from ._types import LiteralRuntime
    from ._types import P

    ServeCommand = t.Literal["serve", "serve-grpc"]
    OutputLiteral = t.Literal["json", "pretty", "porcelain"]

    TupleStr = tuple[str, ...]
else:
    TupleStr = tuple
    torch = LazyLoader("torch", globals(), "torch")


# NOTE: We need to do this so that overload can register
# correct overloads to typing registry
if sys.version_info[:2] >= (3, 11):
    from typing import overload
else:
    from typing_extensions import overload

if sys.version_info[:2] >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


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


def parse_device_callback(
    ctx: click.Context, param: click.Parameter, value: tuple[tuple[str], ...] | None
) -> TupleStr | None:
    if value is None:
        return value

    if not LazyType(TupleStr).isinstance(value):
        ctx.fail(f"{param} only accept multiple values, not {type(value)} (value: {value})")

    el: TupleStr = tuple(i for k in value for i in k)

    # NOTE: --device all is a special case
    if len(el) == 1 and el[0] == "all":
        return tuple(map(str, gpu_count()))

    return el


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


def model_id_option(factory: t.Any, model_env: EnvVarMixin | None = None):
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


def workers_per_resource_option(factory: t.Any, build: bool = False):
    help_str = """Number of workers per resource assigned.
    See https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy
    for more information. By default, this is set to 1.

    **Note**: ``--workers-per-resource`` will also accept the following strategies:

    - ``round_robin``: Similar behaviour when setting ``--workers-per-resource 1``. This is useful for smaller models.

    - ``conserved``: This will determine the number of available GPU resources, and only assign one worker for the LLMRunner. For example, if ther are 4 GPUs available, then ``conserved`` is equivalent to ``--workers-per-resource 0.25``.
    """
    if build:
        help_str += """\n
    **Note**: The workers value passed into 'build' will determine how the LLM can
    be provisioned in Kubernetes as well as in standalone container. This will
    ensure it has the same effect with 'openllm start --workers ...'"""
    return factory.option("--workers-per-resource", default=None, help=help_str, type=str, required=False)


def quantize_option(factory: t.Any, build: bool = False, model_env: EnvVarMixin | None = None):
    envvar = None
    if model_env is not None:
        envvar = model_env.quantize
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
        envvar=envvar,
        show_envvar=True if envvar is not None else False,
    )


def bettertransformer_option(factory: t.Any, build: bool = False, model_env: EnvVarMixin | None = None):
    envvar = None
    if model_env is not None:
        envvar = model_env.bettertransformer
    return factory.option(
        "--bettertransformer",
        is_flag=True,
        default=None,
        help="Apply FasterTransformer wrapper to serve model. This will applies during serving time."
        if not build
        else "Set defaul environment variable whether to serve this model with FasterTransformer in build time.",
        envvar=envvar,
        show_envvar=True if envvar is not None else False,
    )


_adapter_mapping_key = "adapter_map"


def _id_callback(ctx: click.Context, _: click.Parameter, value: tuple[str, ...] | None) -> dict[str, str] | None:
    if not value:
        return
    if _adapter_mapping_key not in ctx.params:
        ctx.params[_adapter_mapping_key] = {}
    for v in value:
        adapter_id, *adapter_name = v.rsplit(":", maxsplit=1)
        try:
            # try to resolve the full path if users pass in relative,
            # currently only support one level of resolve path.
            adapter_id = resolve_user_filepath(adapter_id, os.getcwd())
        except FileNotFoundError:
            pass
        ctx.params[_adapter_mapping_key][adapter_id] = adapter_name[0] if len(adapter_name) > 0 else None


class OpenLLMCommandGroup(BentoMLCommandGroup):
    NUMBER_OF_COMMON_PARAMS = 4  # parameters in common_params + 1 faked group option header

    @staticmethod
    def common_params(f: AnyCallable):
        """This is not supposed to be used with unprocessed click function.

        This should be used a the last currying from common_params -> usage_tracking -> exception_handling.
        """
        # The following logics is similar to one of BentoMLCommandGroup

        from bentoml._internal.configuration import DEBUG_ENV_VAR
        from bentoml._internal.configuration import QUIET_ENV_VAR

        @cog.optgroup.group("Miscellaneous options")
        @cog.optgroup.option(
            "-q", "--quiet", envvar=QUIET_ENV_VAR, is_flag=True, default=False, help="Suppress all output."
        )
        @cog.optgroup.option(
            "--debug", "--verbose", envvar=DEBUG_ENV_VAR, is_flag=True, default=False, help="Print out debug logs."
        )
        @cog.optgroup.option(
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

        return wrapper

    @staticmethod
    def usage_tracking(func: AnyCallable, group: click.Group, **attrs: t.Any) -> AnyCallable:
        """This is not supposed to be used with unprocessed click function.

        This should be used a the last currying from common_params -> usage_tracking -> exception_handling.
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

        return wrapper

    @staticmethod
    def exception_handling(func: AnyCallable, group: click.Group, **attrs: t.Any) -> ClickFunctionWrapper[..., t.Any]:
        """This is not supposed to be used with unprocessed click function.

        This should be used a the last currying from common_params -> usage_tracking -> exception_handling.
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
                # support start from a bento
                try:
                    return start_command_factory(bentoml.get(cmd_name), _context_settings=_CONTEXT_SETTINGS)
                except bentoml.exceptions.NotFound:
                    pass
                raise click.BadArgumentUsage(
                    f"{cmd_name} is not a valid model identifier supported by OpenLLM."
                ) from None
        elif ctx.command.name == "start-grpc":
            try:
                return _cached_grpc[cmd_name]
            except KeyError:
                # support start from a bento
                try:
                    return start_command_factory(
                        bentoml.get(cmd_name), _context_settings=_CONTEXT_SETTINGS, _serve_grpc=True
                    )
                except bentoml.exceptions.NotFound:
                    pass
                raise click.BadArgumentUsage(
                    f"{cmd_name} is not a valid model identifier supported by OpenLLM."
                ) from None
        return super().get_command(ctx, cmd_name)

    def list_commands(self, ctx: click.Context) -> list[str]:
        if ctx.command.name == "start" or ctx.command.name == "start-grpc":
            return list(openllm.CONFIG_MAPPING.keys())

        return super().list_commands(ctx)

    @override
    def command(self, *args: t.Any, **attrs: t.Any):
        """Override the default 'cli.command' with supports for aliases for given command, and it wraps the implementation with common parameters."""
        if "context_settings" not in attrs:
            attrs["context_settings"] = {}
        if "max_content_width" not in attrs["context_settings"]:
            attrs["context_settings"]["max_content_width"] = 120
        aliases = attrs.pop("aliases", None)

        def wrapper(f: AnyCallable) -> click.Command:
            name = f.__name__.lower()
            if name.endswith("_command"):
                name = name[:-8]
            name = name.replace("_", "-")
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


@click.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="openllm")
@click.version_option(__version__, "--version", "-v")
def cli():
    """\b
     ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
    ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
    ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
    ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
    ╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
     ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝.

    \b
    An open platform for operating large language models in production.
    Fine-tune, serve, deploy, and monitor any LLMs with ease.
    """  # noqa: D205


@cli.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="start", aliases=["start-http"])
def start_command():
    """Start any LLM as a REST server.

    \b
    ```bash
    $ openllm <start|start-http> <model_name> --<options> ...
    ```
    """


@cli.group(cls=OpenLLMCommandGroup, context_settings=_CONTEXT_SETTINGS, name="start-grpc")
def start_grpc_command():
    """Start any LLM as a gRPC server.

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
    """Parsing `bentoml serve|serve-grpc` click.Option to be parsed via `openllm start`."""
    from bentoml_cli.cli import cli

    command = "serve" if not serve_grpc else "serve-grpc"
    group = cog.optgroup.group(
        f"Start a {'HTTP' if not serve_grpc else 'gRPC'} server options",
        help=f"Related to serving the model [synonymous to `bentoml {'serve-http' if not serve_grpc else command }`]",
    )

    def decorator(f: t.Callable[t.Concatenate[int, str | None, P], openllm.LLMConfig]):
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
            f = cog.optgroup.option(*param_decls, **attrs)(f)

        return group(f)

    return decorator


_http_server_args = parse_serve_args(False)
_grpc_server_args = parse_serve_args(True)


def start_decorator(llm_config: openllm.LLMConfig, serve_grpc: bool = False):
    opts = [
        llm_config.to_click_options,
        _http_server_args if not serve_grpc else _grpc_server_args,
        cog.optgroup.group("General LLM Options", help="The following options are related to running the LLM Server."),
        cog.optgroup.option(
            "--server-timeout",
            type=int,
            default=None,
            help="Server timeout in seconds",
        ),
        workers_per_resource_option(cog.optgroup),
        model_id_option(cog.optgroup, model_env=llm_config["env"]),
        cog.optgroup.option(
            "--fast",
            is_flag=True,
            default=False,
            help="Bypass auto model checks and setup. This option is ahead-of-serving time.",
        ),
        cog.optgroup.group(
            "LLM Optimization Options",
            help="""\
    These options are related for dynamic optimization on the fly. Current supported strategies:

    - int8: Quantize the model with 8bit (bitsandbytes required)

    - int4: Quantize the model with 4bit (bitsandbytes required)

    - bettertransformer: Convert given model to FastTransformer

    The following are currently being worked on:

    - GPTQ: [paper](https://arxiv.org/abs/2210.17323)

    - DeepSpeed Inference: [link](https://www.deepspeed.ai/inference/)

      """,
        ),
        cog.optgroup.option(
            "--device",
            type=dantic.CUDA,
            multiple=True,
            envvar="CUDA_VISIBLE_DEVICES",
            callback=parse_device_callback,
            help=f"Assign GPU devices (if available) for {llm_config['model_name']}.",
            show_envvar=True,
        ),
        cog.optgroup.option(
            "--runtime",
            type=click.Choice(["ggml", "transformers"]),
            default="transformers",
            help="The runtime to use for the given model. Default is transformers.",
        ),
        quantize_option(cog.optgroup, model_env=llm_config["env"]),
        bettertransformer_option(cog.optgroup, model_env=llm_config["env"]),
        cog.optgroup.group(
            "Fine-tuning related options",
            help="""\
    Note that the argument `--adapter-id` can accept the following format:

    - `--adapter-id /path/to/adapter` (local adapter)

 j   - `--adapter-id remote/adapter` (remote adapter from HuggingFace Hub)

    - `--adapter-id remote/adapter:eng_lora` (two previous adapter options with the given adapter_name)

    ```bash

    $ openllm start opt --adapter-id /path/to/adapter_dir --adapter-id remote/adapter:eng_lora

    ```
    """,
        ),
        cog.optgroup.option(
            "--adapter-id",
            default=None,
            help="Optional name or path for given LoRA adapter" + f" to wrap '{llm_config['model_name']}'",
            multiple=True,
            callback=_id_callback,
            metavar="[PATH | [remote/][adapter_name:]adapter_id][, ...]",
        ),
        click.option("--return-process", is_flag=True, default=False, help="Internal use only.", hidden=True),
    ]

    def decorator(f: AnyCallable) -> AnyCallable:
        for opt in reversed(opts):
            f = opt(f)
        return f

    return decorator


def parse_config_options(
    config: openllm.LLMConfig,
    server_timeout: int,
    workers_per_resource: float,
    device: tuple[str, ...] | None,
    environ: DictStrAny,
) -> DictStrAny:
    _bentoml_config_options_env = environ.pop("BENTOML_CONFIG_OPTIONS", "")
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

    _bentoml_config_options_env += " " if _bentoml_config_options_env else "" + " ".join(_bentoml_config_options_opts)
    environ["BENTOML_CONFIG_OPTIONS"] = _bentoml_config_options_env
    return environ


def start_command_factory(
    model_name_or_bento: str | bentoml.Bento,
    _context_settings: DictStrAny | None = None,
    _serve_grpc: bool = False,
) -> click.Command:
    """Generate a 'click.Command' for any given LLM.

    Args:
        model_name_or_bento: The name of the model or the ``bentoml.Bento`` instance.

    Returns:
        The click.Command for starting the model server

    Note that the internal commands will return the llm_config and a boolean determine
    whether the server is run with GPU or not.
    """
    group = start_command if not _serve_grpc else start_grpc_command

    if isinstance(model_name_or_bento, bentoml.Bento):
        if "start_name" not in model_name_or_bento.info.labels:
            raise click.BadOptionUsage(
                "model_name",
                f"'{model_name_or_bento.tag}' is built with older version of OpenLLM and not supported with 'openllm start'. Please use 'bentoml {'serve-http' if not _serve_grpc else 'serve-grpc'} {model_name_or_bento.tag!s}' instead.",
            )
        llm_config = openllm.AutoConfig.infer_class_from_name(
            model_name_or_bento.info.labels["start_name"]
        ).model_construct_json(model_name_or_bento.info.labels["configuration"])
        return start_bento(group, model_name_or_bento, llm_config, _serve_grpc, context_settings=_CONTEXT_SETTINGS)
    else:
        llm_config = openllm.AutoConfig.for_model(model_name_or_bento)

        return start_model(
            group,
            model_name_or_bento,
            llm_config,
            _serve_grpc,
            name=llm_config["model_name"],
            context_settings=_context_settings or {},
            short_help=f"Start a LLMServer for '{model_name_or_bento}'",
            aliases=[llm_config["start_name"]] if llm_config["name_type"] == "dasherize" else None,
            help=f"""\
{llm_config['env'].start_docstring}

\b
Available model_id(s): {llm_config['model_ids']} [default: {llm_config['default_id']}]
""",
        )


def start_bento_docstring(bento: bentoml.Bento, llm_config: openllm.LLMConfig, serve_grpc: bool):
    environ = parse_config_options(llm_config, llm_config["timeout"], llm_config["workers_per_resource"], None, {})

    serve_cmd_envvar = {
        "OPENLLM_MODEL_ID": f"$(bentoml models get {bento.info.labels['_framework']}-{bento.info.labels['_type']} -o path)",
        "BENTOML_DEBUG": get_debug_mode(),
        "BENTOML_CONFIG_OPTIONS": environ["BENTOML_CONFIG_OPTIONS"],
    }

    return f"""\
Start {bento!r} with OpenLLM.

\b
This is a lightwrapper around 'bentoml serve' to provide nicer interaction with LLM Bentos.

\b
The equivalent 'bentoml {'serve' if not serve_grpc else 'serve-grpc'}' command:

\b
```bash
$ {' '.join([f'{k}={v}' for k, v in serve_cmd_envvar.items()])} bentoml {'serve-http' if not serve_grpc else 'serve-grpc'} {bento.tag!s}
```

\b
> Note that if you want to enable GPU with 'bentoml serve', add the following to BENTOML_CONFIG_OPTIONS:

\b
If you have more than 1 GPU:

\b
```bash
BENTOML_CONFIG_OPTIONS += ' runners."llm-{llm_config['start_name']}-runner".resources."nvidia.com/gpu"[<gpu_idx>]=<gpu_device_id>'
```

Make sure to adjust `workers_per_resource` in BENTOML_CONFIG_OPTIONS accordingly. See https://docs.bentoml.com/en/latest/guides/scheduling.html
for more information.

\b
If you only have 1 GPU:

\b
```bash
BENTOML_CONFIG_OPTIONS += ' runners."llm-{llm_config['start_name']}-runner".resources."nvidia.com/gpu"=[<gpu_device_id>]'
```
"""


def noop_command(
    group: click.Group, llm_config: openllm.LLMConfig, reason: str, **command_attrs: t.Any
) -> click.Command:
    context_settings = command_attrs.pop("context_settings", {})
    context_settings["ignore_unknown_options"] = True
    context_settings["allow_extra_args"] = True
    command_attrs["context_settings"] = context_settings

    # NOTE: The model requires GPU, therefore we will return a dummy command
    @group.command(**command_attrs)
    def noop(**_: t.Any) -> openllm.LLMConfig:
        _echo(reason, fg="red")
        analytics.track_start_init(llm_config)
        return llm_config

    return noop


def prerequisite_check(
    ctx: click.Context,
    llm_config: openllm.LLMConfig,
    env: EnvVarMixin,
    gpu_available: tuple[str, ...],
    quantize: t.LiteralString | None,
    adapter_map: dict[str, str | None] | None,
    num_workers: int,
) -> None:
    if get_debug_mode():
        _echo("Running prerequisite check.", fg="magenta")
    if quantize:
        if len(gpu_available) < 1:
            _echo(f"Quantization requires at least 1 GPU (got {len(gpu_available)})", fg="red")
            ctx.exit(1)
        if env.framework_value != "pt":
            _echo("Quantization is currently only available for PyTorch models.", fg="red")
            ctx.exit(1)

    if adapter_map and not is_peft_available():
        _echo(
            "Using adapter requires 'peft' to be available. Make sure to install with 'pip install \"openllm[fine-tune]\"'",
            fg="red",
        )
        ctx.exit(1)

    requirements = llm_config["requirements"]
    if requirements is not None and len(requirements) > 0:
        missing_requirements = [i for i in requirements if importlib.util.find_spec(i) is None]
        if len(missing_requirements) > 0:
            _echo(
                f"Make sure to have the following dependencies available: {missing_requirements}",
                fg="yellow",
            )

    if num_workers > 1 and len(gpu_available) < num_workers:
        raise click.BadOptionUsage(
            "workers_per_resource",
            f"# of workers is infered to {num_workers} GPUs per runner worker, while there are only"
            f"'{gpu_available}' for inference. (Tip: Try again using '--workers-per-resource={1/len(gpu_available)}')",
            ctx=ctx,
        )


_wpr_strategies = {"round_robin", "conserved"}


def start_bento(
    group: click.Group,
    bento: bentoml.Bento,
    llm_config: openllm.LLMConfig,
    serve_grpc: bool,
    **command_attrs: t.Any,
):
    gpu_available = gpu_count()
    if llm_config["requires_gpu"] and len(gpu_available) < 1:
        return noop_command(
            group, llm_config, f"No GPU available, while {bento!r} requires GPU to run.", **command_attrs
        )

    command_attrs["help"] = start_bento_docstring(bento, llm_config, serve_grpc)

    # Now we have to format the model_id accordingly based on the model_fs
    model_type = bento.info.labels["_type"]
    model_framework = bento.info.labels["_framework"]
    # the models should have the type
    try:
        model_store = ModelStore(bento._fs.opendir("models"))
        model = model_store.get(f"{model_framework}-{model_type}")
    except fs.errors.ResourceNotFound:
        # new behaviour with BentoML models
        _model_store = BentoMLContainer.model_store.get()
        model = _model_store.get(f"{model_framework}-{model_type}")
    except bentoml.exceptions.NotFound:
        raise OpenLLMException(f"Failed to find models for {llm_config['start_name']}") from None

    @group.command(**command_attrs)
    @start_decorator(llm_config, serve_grpc=serve_grpc)
    @click.pass_context
    def start_cmd(
        ctx: click.Context,
        server_timeout: int | None,
        model_id: str | None,
        workers_per_resource: t.LiteralString | float | None,
        device: tuple[str, ...] | None,
        quantize: t.Literal["int8", "int4", "gptq"] | None,
        bettertransformer: bool | None,
        runtime: t.Literal["ggml", "transformers"],
        fast: bool,
        adapter_id: str | None,
        return_process: bool,
        **attrs: t.Any,
    ) -> openllm.LLMConfig | subprocess.Popen[bytes]:
        if model_id is not None:
            _echo("'model_id' has no effect when starting a BentoLLM", fg="yellow")

        adapter_map: dict[str, str | None] | None = attrs.pop(_adapter_mapping_key, None)

        config, server_attrs = llm_config.model_validate_click(**attrs)
        server_timeout = first_not_none(server_timeout, default=config["timeout"])

        server_attrs.update({"working_dir": os.path.dirname(__file__), "timeout": server_timeout})
        if serve_grpc:
            server_attrs["grpc_protocol_version"] = "v1"
        # NOTE: currently, theres no development args in bentoml.Server. To be fixed upstream.
        development = server_attrs.pop("development")
        server_attrs.setdefault("production", not development)

        workers_per_resource = first_not_none(workers_per_resource, default=config["workers_per_resource"])

        if isinstance(workers_per_resource, str):
            if workers_per_resource == "round_robin":
                workers_per_resource = 1.0
            elif workers_per_resource == "conserved":
                if device:
                    available_gpu = device
                else:
                    available_gpu = gpu_count()
                if len(available_gpu) != 0:
                    workers_per_resource = float(1 / len(available_gpu))
                else:
                    workers_per_resource = 1.0
            else:
                try:
                    workers_per_resource = float(workers_per_resource)
                except ValueError:
                    ctx.fail(f"'workers_per_resource' only accept '{_wpr_strategies}' as possible strategies.")

        num_workers = int(1 / workers_per_resource)

        # Create a new model env to work with the envvar during CLI invocation
        env = EnvVarMixin(
            config["model_name"], bettertransformer=bettertransformer, quantize=quantize, runtime=runtime
        )

        prerequisite_check(ctx, config, env, gpu_available, quantize, adapter_map, num_workers)

        # NOTE: This is to set current configuration
        start_env = os.environ.copy()
        start_env = parse_config_options(config, server_timeout, workers_per_resource, device, start_env)

        if fast:
            _echo(f"Fast mode has no effects when 'start' {bento.tag!s}", fg="yellow")

        start_env.update(
            {
                env.framework: env.framework_value,
                env.config: config.model_dump_json().decode(),
                env.runtime: env.runtime_value,
                "BENTOML_DEBUG": str(get_debug_mode()),
                "BENTOML_HOME": os.environ.get("BENTOML_HOME", BentoMLContainer.bentoml_home.get()),
                "OPENLLM_MODEL_ID": model.path,
            }
        )

        if adapter_map:
            _echo(f"OpenLLM will convert '{bento.tag!s}' to use provided adapters layers: {list(adapter_map)}")
        start_env["OPENLLM_ADAPTER_MAP"] = orjson.dumps(adapter_map).decode()

        if bettertransformer is not None:
            start_env[env.bettertransformer] = bettertransformer
        if quantize is not None:
            start_env[env.quantize] = quantize

        if serve_grpc:
            server = bentoml.GrpcServer(bento, **server_attrs)
        else:
            server = bentoml.HTTPServer(bento, **server_attrs)
        analytics.track_start_init(config)

        server.start(env=start_env, text=True)
        process = server.process
        assert process

        if return_process:
            return process

        try:
            assert process.stdout
            with process:
                for line in iter(process.stdout.readline, b""):
                    _echo(line.strip(), fg="white")
        except Exception as err:
            _echo(f"Error caught while starting LLM Server:\n{err}", fg="red")
            raise

        # NOTE: Return the configuration for telemetry purposes.
        return config

    return start_cmd


def start_model(
    group: click.Group,
    model_name: str,
    llm_config: openllm.LLMConfig,
    serve_grpc: bool,
    **command_attrs: t.Any,
):
    gpu_available = gpu_count()
    if llm_config["requires_gpu"] and len(gpu_available) < 1:
        # NOTE: The model requires GPU, therefore we will return a dummy command
        command_attrs.update(
            {
                "short_help": "(Disabled because there is no GPU available)",
                "help": f"""{model_name} is currently not available to run on your
                local machine because it requires GPU for inference.""",
            }
        )
        return noop_command(group, llm_config, "No GPU available, therefore this command is disabled", **command_attrs)

    @group.command(**command_attrs)
    @start_decorator(llm_config, serve_grpc=serve_grpc)
    @click.pass_context
    def start_cmd(
        ctx: click.Context,
        server_timeout: int | None,
        model_id: str | None,
        workers_per_resource: str | float | None,
        device: tuple[str, ...] | None,
        quantize: t.Literal["int8", "int4", "gptq"] | None,
        bettertransformer: bool | None,
        runtime: t.Literal["ggml", "transformers"],
        fast: bool,
        adapter_id: str | None,
        return_process: bool,
        **attrs: t.Any,
    ) -> openllm.LLMConfig | subprocess.Popen[bytes]:
        adapter_map: dict[str, str | None] | None = attrs.pop(_adapter_mapping_key, None)

        config, server_attrs = llm_config.model_validate_click(**attrs)
        server_timeout = first_not_none(server_timeout, default=config["timeout"])

        server_attrs.update({"working_dir": os.path.dirname(__file__), "timeout": server_timeout})
        if serve_grpc:
            server_attrs["grpc_protocol_version"] = "v1"
        # NOTE: currently, theres no development args in bentoml.Server. To be fixed upstream.
        development = server_attrs.pop("development")
        server_attrs.setdefault("production", not development)

        workers_per_resource = first_not_none(workers_per_resource, default=config["workers_per_resource"])

        if isinstance(workers_per_resource, str):
            if workers_per_resource == "round_robin":
                workers_per_resource = 1.0
            elif workers_per_resource == "conserved":
                if device:
                    available_gpu = device
                else:
                    available_gpu = gpu_count()
                if len(available_gpu) != 0:
                    workers_per_resource = float(1 / len(available_gpu))
                else:
                    workers_per_resource = 1.0
            else:
                try:
                    workers_per_resource = float(workers_per_resource)
                except ValueError:
                    ctx.fail(f"'workers_per_resource' only accept '{_wpr_strategies}' as possible strategies.")

        num_workers = int(1 / workers_per_resource)

        # Create a new model env to work with the envvar during CLI invocation
        env = EnvVarMixin(
            config["model_name"], bettertransformer=bettertransformer, quantize=quantize, runtime=runtime
        )

        prerequisite_check(ctx, config, env, gpu_available, quantize, adapter_map, num_workers)

        # NOTE: This is to set current configuration
        start_env = os.environ.copy()
        start_env = parse_config_options(config, server_timeout, workers_per_resource, device, start_env)

        if fast and not get_quiet_mode():
            _echo(
                f"Fast mode is enabled. Make sure to download the model before 'start': 'openllm download {model_name}{'--model-id ' + model_id if model_id else ''}'",
                fg="yellow",
            )

        start_env.update(
            {
                env.framework: env.framework_value,
                "BENTOML_DEBUG": str(get_debug_mode()),
                "BENTOML_HOME": os.environ.get("BENTOML_HOME", BentoMLContainer.bentoml_home.get()),
            }
        )

        if adapter_map:
            _echo(f"OpenLLM will convert '{model_name}' to use provided adapters layers: {list(adapter_map)}")

        llm = openllm.infer_auto_class(env.framework_value).for_model(
            model_name,
            model_id=model_id,
            llm_config=config,
            ensure_available=not fast,
            return_runner_kwargs=False,
            quantize=quantize,
            bettertransformer=bettertransformer,
            adapter_map=adapter_map,
            runtime=runtime,
        )

        start_env.update(
            {
                env.config: llm.config.model_dump_json().decode(),
                env.runtime: env.runtime_value,
                "OPENLLM_MODEL": model_name,
                "OPENLLM_MODEL_ID": llm.model_id,
                "OPENLLM_ADAPTER_MAP": orjson.dumps(adapter_map).decode(),
            }
        )

        if bettertransformer is not None:
            start_env[env.bettertransformer] = bettertransformer
        if quantize is not None:
            start_env[env.quantize] = quantize

        if serve_grpc:
            server = bentoml.GrpcServer("_service.py:svc", **server_attrs)
        else:
            server = bentoml.HTTPServer("_service.py:svc", **server_attrs)
        analytics.track_start_init(llm.config)

        server.start(env=start_env, text=True)
        process = server.process
        assert process

        if return_process:
            return process

        try:
            assert process.stdout
            with process:
                for line in iter(process.stdout.readline, b""):
                    _echo(line.strip(), fg="white")
        except Exception as err:
            _echo(f"Error caught while starting LLM Server:\n{err}", fg="red")
            raise
        finally:
            cmd_name = f"openllm build {model_name}"
            if adapter_map is not None:
                cmd_name += " " + " ".join(
                    [
                        f"--adapter-id {s}"
                        for s in [
                            f"{p}:{name}" if name not in (None, "default") else p for p, name in adapter_map.items()
                        ]
                    ]
                )
            _echo(
                f"\n🚀 Next step: run '{cmd_name}' to create a Bento for {model_name}",
                fg="blue",
            )

        # NOTE: Return the configuration for telemetry purposes.
        return config

    return start_cmd


@cli.command(name="import", aliases=["download"])
@click.argument(
    "model",
    type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()]),
)
@click.argument("model_id", type=click.STRING, default=None, metavar="model_id", required=False)
@click.option(
    "--model-version",
    type=click.STRING,
    default=None,
    help="Optional model version to save for this model. It will be inferred automatically from model-id.",
)
@click.option(
    "--runtime",
    type=click.Choice(["ggml", "transformers"]),
    default="transformers",
    help="The runtime to use for the given model. Default is transformers.",
)
@output_option
@quantize_option(click)
@click.option("--machine", is_flag=True, default=False, hidden=True)
@click.option("--implementation", type=click.Choice(["pt", "tf", "flax"]), default=None, hidden=True)
def download_models_command(
    model: str,
    model_id: str | None,
    model_version: str | None,
    output: OutputLiteral,
    runtime: t.Literal["ggml", "transformers"],
    machine: bool,
    implementation: LiteralRuntime | None,
    quantize: t.Literal["int8", "int4", "gptq"] | None,
):
    """Setup LLM interactively.

    It accepts two positional arguments: `model_name` and `model_id`. The first name determine
    the model type to download, and the second one is the optional model id to download.

    \b
    This `model_id` can be either pretrained model id that you can get from HuggingFace Hub, or
    a custom model path from your custom pretrained model. Note that the custom model path should
    contain all files required to construct `transformers.PretrainedConfig`, `transformers.PreTrainedModel`
    and `transformers.PreTrainedTokenizer` objects.

    \b
    Note: This is useful for development and setup for fine-tune.
    This will be automatically called when `ensure_available=True` in `openllm.LLM.for_model`

    \b
    ``--model-version`` is an optional option to save the model. Note that
    this is recommended when the model_id is a custom path. Usually, if you are only using pretrained
    model from HuggingFace Hub, you don't need to specify this. If this is not specified, we will calculate
    the hash from the last modified time from this custom path

    \b
    ```bash
    $ openllm download opt facebook/opt-2.7b
    ```

    \b
    > If ``quantize`` is passed, the model weights will be saved as quantized weights. You should
    > only use this option if you want the weight to be quantized by default. Note that OpenLLM also
    > support on-demand quantisation during initial startup.
    """
    impl: t.Literal["pt", "tf", "flax"] = first_not_none(implementation, default=EnvVarMixin(model).framework_value)
    llm = openllm.infer_auto_class(impl).for_model(
        model,
        model_id=model_id,
        model_version=model_version,
        runtime=runtime,
        return_runner_kwargs=False,
        quantize=quantize,
        ensure_available=False,
    )

    _previously_saved = False
    try:
        _ref = bentoml.models.get(llm.tag)
        _previously_saved = True
    except bentoml.exceptions.NotFound:
        if not machine and output == "pretty":
            _echo(
                f"'{model}' with 'model_id={model_id}' does not exists in local store. Saving to store...",
                fg="yellow",
                nl=True,
            )

        _ref = llm.import_model(trust_remote_code=llm.__llm_trust_remote_code__)

    if not machine:
        if output == "pretty":
            if _previously_saved:
                _echo(
                    f"{model} with 'model_id={model_id}' is already setup for framework '{impl}': {_ref.tag!s}",
                    nl=True,
                    fg="yellow",
                )
            else:
                _echo(f"Saved model: {_ref.tag}")
        elif output == "json":
            _echo(
                orjson.dumps(
                    {"previously_setup": _previously_saved, "framework": impl, "tag": str(_ref.tag)},
                    option=orjson.OPT_INDENT_2,
                ).decode()
            )
        else:
            _echo(_ref.tag)
    if is_torch_available() and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return _ref


_cached_http = {key: start_command_factory(key, _context_settings=_CONTEXT_SETTINGS) for key in openllm.CONFIG_MAPPING}
_cached_grpc = {
    key: start_command_factory(key, _context_settings=_CONTEXT_SETTINGS, _serve_grpc=True)
    for key in openllm.CONFIG_MAPPING
}


@overload
def _start(
    model_name: str | bentoml.Bento,
    /,
    model_id: str | None = ...,
    timeout: int = ...,
    workers_per_resource: t.Literal["conserved", "round_robin"] | float | None = ...,
    device: tuple[str, ...] | t.Literal["all"] | None = ...,
    quantize: t.Literal["int8", "int4", "gptq"] | None = ...,
    bettertransformer: bool | None = ...,
    runtime: t.Literal["ggml", "transformers"] = ...,
    fast: bool = ...,
    adapter_map: dict[t.LiteralString, str | None] | None = ...,
    framework: t.Literal["flax", "tf", "pt"] | None = ...,
    additional_args: ListStr | None = ...,
    _serve_grpc: bool = ...,
    __test__: t.Literal[False] = ...,
) -> openllm.LLMConfig:
    ...


@overload
def _start(
    model_name: str | bentoml.Bento,
    /,
    model_id: str | None = ...,
    timeout: int = ...,
    workers_per_resource: t.Literal["conserved", "round_robin"] | float | None = ...,
    device: tuple[str, ...] | t.Literal["all"] | None = ...,
    quantize: t.Literal["int8", "int4", "gptq"] | None = ...,
    bettertransformer: bool | None = ...,
    runtime: t.Literal["ggml", "transformers"] = ...,
    fast: bool = ...,
    adapter_map: dict[t.LiteralString, str | None] | None = ...,
    framework: t.Literal["flax", "tf", "pt"] | None = ...,
    additional_args: ListStr | None = ...,
    _serve_grpc: bool = ...,
    __test__: t.Literal[True] = ...,
) -> subprocess.Popen[bytes]:
    ...


def _start(
    model_name: str | bentoml.Bento,
    /,
    model_id: str | None = None,
    timeout: int = 30,
    workers_per_resource: t.Literal["conserved", "round_robin"] | float | None = None,
    device: tuple[str, ...] | t.Literal["all"] | None = None,
    quantize: t.Literal["int8", "int4", "gptq"] | None = None,
    bettertransformer: bool | None = None,
    runtime: t.Literal["ggml", "transformers"] = "transformers",
    fast: bool = False,
    adapter_map: dict[t.LiteralString, str | None] | None = None,
    framework: t.Literal["flax", "tf", "pt"] | None = None,
    additional_args: ListStr | None = None,
    _serve_grpc: bool = False,
    __test__: bool = False,
) -> openllm.LLMConfig | subprocess.Popen[bytes]:
    """Python API to start a LLM server. These provides one-to-one mapping to CLI arguments.

    For all additional arguments, pass it as string to ``additional_args``. For example, if you want to
    pass ``--port 5001``, you can pass ``additional_args=["--port", "5001"]``

    > **Note**: This will create a blocking process, so if you use this API, you can create a running sub thread
    > to start the server instead of blocking the main thread.

    ``openllm.start`` will invoke ``click.Command`` under the hood, so it behaves exactly the same as the CLI interaction.

    > **Note**: ``quantize`` and ``bettertransformer`` are mutually exclusive.

    Args:
        model_name: The model name to start this LLM
        model_id: Optional model id for this given LLM
        timeout: The server timeout
        workers_per_resource: Number of workers per resource assigned.
                              See https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy
                              for more information. By default, this is set to 1.

                              > **Note**: ``--workers-per-resource`` will also accept the following strategies:

                              > - ``round_robin``: Similar behaviour when setting ``--workers-per-resource 1``. This is useful for smaller models.

                              > - ``conserved``: Thjis will determine the number of available GPU resources, and only assign
                                                 one worker for the LLMRunner. For example, if ther are 4 GPUs available, then ``conserved`` is
                                                 equivalent to ``--workers-per-resource 0.25``.
        device: Assign GPU devices (if available) to this LLM. By default, this is set to ``None``. It also accepts 'all'
                argument to assign all available GPUs to this LLM.
        quantize: Quantize the model weights. This is only applicable for PyTorch models.
                  Possible quantisation strategies:
                  - int8: Quantize the model with 8bit (bitsandbytes required)
                  - int4: Quantize the model with 4bit (bitsandbytes required)
                  - gptq: Quantize the model with GPTQ (autogptq required)
        bettertransformer: Convert given model to FastTransformer with PyTorch.
        runtime: The runtime to use for this LLM. By default, this is set to ``transformers``. In the future, this will include supports for GGML.
        fast: Enable fast mode. This will skip downloading models, and will raise errors if given model_id does not exists under local store.
        adapter_map: The adapter mapping of LoRA to use for this LLM. It accepts a dictionary of ``{adapter_id: adapter_name}``.
        framework: The framework to use for this LLM. By default, this is set to ``pt``.
        additional_args: Additional arguments to pass to ``openllm start``.
    """
    if isinstance(model_name, str):
        _ModelEnv = EnvVarMixin(model_name)
        if framework is None:
            framework = _ModelEnv.framework_value
        os.environ[_ModelEnv.framework] = framework

    args: ListStr = ["--runtime", runtime]
    if model_id:
        if isinstance(model_id, bentoml.Bento):
            logger.warning("'model_id' has no effect if since %s is already a Bento.", model_name)
        else:
            args.extend(["--model-id", model_id])
    if timeout:
        args.extend(["--server-timeout", str(timeout)])
    if workers_per_resource:
        args.extend(
            [
                "--workers-per-resource",
                str(workers_per_resource) if not isinstance(workers_per_resource, str) else workers_per_resource,
            ]
        )
    if device and not os.getenv("CUDA_VISIBLE_DEVICES"):
        args.extend(["--device", ",".join(device)])

    if quantize and bettertransformer:
        raise OpenLLMException("'quantize' and 'bettertransformer' are currently mutually exclusive.")

    if quantize:
        args.extend(["--quantize", str(quantize)])
    if bettertransformer:
        args.append("--bettertransformer")
    if fast:
        args.append("--fast")
    if adapter_map:
        args.extend(
            list(
                itertools.chain.from_iterable(
                    [["--adapter-id", f"{k}{':'+v if v else ''}"] for k, v in adapter_map.items()]
                )
            )
        )
    if additional_args:
        args.extend(additional_args)

    if __test__:
        args.append("--return-process")

    return start_command_factory(model_name, _context_settings=_CONTEXT_SETTINGS, _serve_grpc=_serve_grpc).main(
        args=args if len(args) > 0 else None,
        standalone_mode=False,
    )


@overload
def _build(
    model_name: str,
    /,
    *,
    model_id: str | None = ...,
    model_version: str | None = ...,
    quantize: t.Literal["int8", "int4", "gptq"] | None = ...,
    bettertransformer: bool | None = ...,
    adapter_map: dict[str, str | None] | None = ...,
    build_ctx: str | None = ...,
    enable_features: tuple[str, ...] | None = ...,
    workers_per_resource: int | float | None = ...,
    runtime: t.Literal["ggml", "transformers"] = ...,
    dockerfile_template: str | None = ...,
    overwrite: bool = ...,
    format: t.Literal["bento"] = "bento",
    additional_args: list[str] | None = ...,
) -> bentoml.Bento:
    ...


@overload
def _build(
    model_name: str,
    /,
    *,
    model_id: str | None = ...,
    model_version: str | None = ...,
    quantize: t.Literal["int8", "int4", "gptq"] | None = ...,
    bettertransformer: bool | None = ...,
    adapter_map: dict[str, str | None] | None = ...,
    build_ctx: str | None = ...,
    enable_features: tuple[str, ...] | None = ...,
    workers_per_resource: int | float | None = ...,
    runtime: t.Literal["ggml", "transformers"] = ...,
    dockerfile_template: str | None = ...,
    overwrite: bool = ...,
    format: t.Literal["container"] = ...,
    additional_args: list[str] | None = ...,
) -> str:
    ...


def _build(
    model_name: str,
    /,
    *,
    model_id: str | None = None,
    model_version: str | None = None,
    quantize: t.Literal["int8", "int4", "gptq"] | None = None,
    bettertransformer: bool | None = None,
    adapter_map: dict[str, str | None] | None = None,
    build_ctx: str | None = None,
    enable_features: tuple[str, ...] | None = None,
    workers_per_resource: int | float | None = None,
    runtime: t.Literal["ggml", "transformers"] = "transformers",
    dockerfile_template: str | None = None,
    overwrite: bool = False,
    format: t.Literal["bento", "container"] = "bento",
    additional_args: list[str] | None = None,
) -> bentoml.Bento | str:
    """Package a LLM into a Bento.

    The LLM will be built into a BentoService with the following structure:
    if ``quantize`` is passed, it will instruct the model to be quantized dynamically during serving time.
    if ``bettertransformer`` is passed, it will instruct the model to apply FasterTransformer during serving time.

    ``openllm.build`` will invoke ``click.Command`` under the hood, so it behaves exactly the same as ``openllm build`` CLI.

    > **Note**: ``quantize`` and ``bettertransformer`` are mutually exclusive.

    Args:
        model_name: The model name to start this LLM
        model_id: Optional model id for this given LLM
        model_version: Optional model version for this given LLM
        quantize: Quantize the model weights. This is only applicable for PyTorch models.
                  Possible quantisation strategies:
                  - int8: Quantize the model with 8bit (bitsandbytes required)
                  - int4: Quantize the model with 4bit (bitsandbytes required)
                  - gptq: Quantize the model with GPTQ (autogptq required)
        bettertransformer: Convert given model to FastTransformer with PyTorch.
        adapter_map: The adapter mapping of LoRA to use for this LLM. It accepts a dictionary of ``{adapter_id: adapter_name}``.
        build_ctx: The build context to use for building BentoLLM. By default, it sets to current directory.
        enable_features: Additional OpenLLM features to be included with this BentoLLM.
        workers_per_resource: Number of workers per resource assigned.
                              See https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy
                              for more information. By default, this is set to 1.

                              > **Note**: ``--workers-per-resource`` will also accept the following strategies:

                              > - ``round_robin``: Similar behaviour when setting ``--workers-per-resource 1``. This is useful for smaller models.

                              > - ``conserved``: This will determine the number of available GPU resources, and only assign
                                                 one worker for the LLMRunner. For example, if ther are 4 GPUs available, then ``conserved`` is
                                                 equivalent to ``--workers-per-resource 0.25``.
        runtime: The runtime to use for this LLM. By default, this is set to ``transformers``. In the future, this will include supports for GGML.
        dockerfile_template: The dockerfile template to use for building BentoLLM. See
                             https://docs.bentoml.com/en/latest/guides/containerization.html#dockerfile-template.
        overwrite: Whether to overwrite the existing BentoLLM. By default, this is set to ``False``.
        format: The output format to build this LLM. By default it will build the BentoLLM. 'container' is equivalent of 'openllm build && bentoml containerize <bento_tag>'
        additional_args: Additional arguments to pass to ``openllm build``.

    Returns:
        ``bentoml.Bento | str``: BentoLLM instance. This can be used to serve the LLM or can be pushed to BentoCloud.
                                 If 'format="container"', then it returns the default 'container_name:container_tag'
    """
    args: ListStr = [model_name, "--runtime", runtime, "--format", format]

    if quantize and bettertransformer:
        raise OpenLLMException("'quantize' and 'bettertransformer' are currently mutually exclusive.")

    if quantize:
        args.extend(["--quantize", quantize])
    if bettertransformer:
        args.append("--bettertransformer")

    if model_id:
        args.extend(["--model-id", model_id])
    if build_ctx:
        args.extend(["--build-ctx", build_ctx])
    if enable_features:
        args.extend([f"--enable-features={f}" for f in enable_features])
    if workers_per_resource:
        args.extend(["--workers-per-resource", str(workers_per_resource)])
    if overwrite:
        args.append("--overwrite")
    if adapter_map:
        args.extend([f"--adapter-id={k}{':'+v if v is not None else ''}" for k, v in adapter_map.items()])
    if model_version:
        args.extend(["--model-version", model_version])
    if dockerfile_template:
        args.extend(["--dockerfile-template", dockerfile_template])
    if additional_args:
        args.extend(additional_args)

    return build_command.main(args=args, standalone_mode=False)


def _import_model(
    model_name: str,
    /,
    *,
    model_id: str | None = None,
    model_version: str | None = None,
    runtime: t.Literal["ggml", "transformers"] = "transformers",
    implementation: LiteralRuntime = "pt",
    quantize: t.Literal["int8", "int4", "gptq"] | None = None,
    additional_args: t.Sequence[str] | None = None,
) -> bentoml.Model:
    """Import a LLM into local store.

    > **Note**: If ``quantize`` is passed, the model weights will be saved as quantized weights. You should
    > only use this option if you want the weight to be quantized by default. Note that OpenLLM also
    > support on-demand quantisation during initial startup.

    ``openllm.download`` will invoke ``click.Command`` under the hood, so it behaves exactly the same as the CLI ``openllm import``.

    > **Note**: ``openllm.start`` will automatically invoke ``openllm.download`` under the hood.

    Args:
        model_name: The model name to start this LLM
        model_id: Optional model id for this given LLM
        model_version: Optional model version for this given LLM
        runtime: The runtime to use for this LLM. By default, this is set to ``transformers``. In the future, this will include supports for GGML.
        implementation: The implementation to use for this LLM. By default, this is set to ``pt``.
        quantize: Quantize the model weights. This is only applicable for PyTorch models.
                  Possible quantisation strategies:
                  - int8: Quantize the model with 8bit (bitsandbytes required)
                  - int4: Quantize the model with 4bit (bitsandbytes required)
                  - gptq: Quantize the model with GPTQ (autogptq required)
        additional_args: Additional arguments to pass to ``openllm import``.

    Returns:
        ``bentoml.Model``:BentoModel of the given LLM. This can be used to serve the LLM or can be pushed to BentoCloud.
    """
    args = [model_name, "--runtime", runtime, "--implementation", implementation, "--machine"]
    if model_id is not None:
        args.append(model_id)
    if model_version is not None:
        args.extend(["--model-version", str(model_version)])
    if additional_args is not None:
        args.extend(additional_args)
    if quantize is not None:
        args.extend(["--quantize", quantize])
    return download_models_command.main(args=args, standalone_mode=False)


def _list_models() -> DictStrAny:
    """List all available models within the local store."""
    args = ["-o", "json", "--show-available", "--machine"]
    return models_command.main(args=args, standalone_mode=False)


start, start_grpc, build, import_model, list_models = (
    codegen.gen_sdk(_start, _serve_grpc=False),
    codegen.gen_sdk(_start, _serve_grpc=True),
    codegen.gen_sdk(_build),
    codegen.gen_sdk(_import_model),
    codegen.gen_sdk(_list_models),
)


@cli.command(context_settings={"token_normalize_func": inflection.underscore})
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
@cog.optgroup.option(
    "--runtime",
    type=click.Choice(["ggml", "transformers"]),
    default="transformers",
    help="The runtime to use for the given model. Default is transformers.",
)
@click.option(
    "--enable-features",
    help="Enable additional features for building this LLM Bento. Available: {}".format(
        ", ".join(openllm.utils.OPTIONAL_DEPENDENCIES)
    ),
    multiple=True,
    nargs=1,
    metavar="FEATURE[,FEATURE]",
)
@click.option(
    "--adapter-id",
    default=None,
    help="Optional adapters id to be included within the Bento. Note that if you are using relative path, '--build-ctx' must be passed.",
    multiple=True,
    metavar="[PATH | [remote/][adapter_name:]adapter_id][, ...]",
)
@click.option("--build-ctx", default=".", help="Build context. This is required if --adapter-id uses relative path")
@click.option(
    "--model-version",
    default=None,
    type=click.STRING,
    help="Model version provided for this 'model-id' if it is a custom path.",
)
@click.option(
    "--dockerfile-template",
    default=None,
    type=click.File(),
    help="Optional custom dockerfile template to be used with this BentoLLM.",
)
@click.option(
    "--format",
    default="bento",
    type=click.Choice(["bento", "container"]),
    help="The output format for 'openllm build'. By default this will build a BentoLLM. 'container' is the shortcut of 'openllm build && bentoml containerize'.",
    hidden=not get_debug_mode(),
)
@click.pass_context
def build_command(
    ctx: click.Context,
    model_name: str,
    model_id: str | None,
    overwrite: bool,
    output: OutputLiteral,
    runtime: t.Literal["ggml", "transformers"],
    quantize: t.Literal["int8", "int4", "gptq"] | None,
    enable_features: tuple[str] | None,
    bettertransformer: bool | None,
    workers_per_resource: float | None,
    adapter_id: tuple[str, ...],
    build_ctx: str | None,
    model_version: str | None,
    dockerfile_template: t.TextIO | None,
    format: t.Literal["bento", "container"],
    **attrs: t.Any,
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
    from bentoml_cli.cli import cli as bentoml_cli

    from ._package import create_bento

    adapter_map: dict[str, str | None] | None = None

    if adapter_id:
        if not build_ctx:
            _echo("'build_ctx' must not be None when '--adapter-id' is passsed.", fg="red")
            ctx.exit(1)

        adapter_map = {}
        for v in adapter_id:
            _adapter_id, *adapter_name = v.rsplit(":", maxsplit=1)
            # We don't resolve full path here, leave it to build
            # we are just doing the parsing here.
            adapter_map[_adapter_id] = adapter_name[0] if len(adapter_name) > 0 else None

    if output == "pretty":
        if overwrite:
            _echo(f"Overwriting existing Bento for {model_name}.", fg="yellow")

    if enable_features:
        enable_features = tuple(itertools.chain.from_iterable((s.split(",") for s in enable_features)))

    _previously_built = False
    current_model_envvar = os.environ.pop("OPENLLM_MODEL", None)
    current_model_id_envvar = os.environ.pop("OPENLLM_MODEL_ID", None)
    current_adapter_map_envvar = os.environ.pop("OPENLLM_ADAPTER_MAP", None)

    llm_config = openllm.AutoConfig.for_model(model_name)

    # NOTE: We set this environment variable so that our service.py logic won't raise RuntimeError
    # during build. This is a current limitation of bentoml build where we actually import the service.py into sys.path
    try:
        os.environ[llm_config["env"].runtime] = runtime
        os.environ["OPENLLM_MODEL"] = inflection.underscore(model_name)
        os.environ["OPENLLM_ADAPTER_MAP"] = orjson.dumps(adapter_map).decode()

        framework_envvar = llm_config["env"].framework_value
        llm = openllm.infer_auto_class(framework_envvar).for_model(
            model_name,
            model_id=model_id,
            llm_config=llm_config,
            quantize=quantize,
            adapter_map=adapter_map,
            bettertransformer=bettertransformer,
            return_runner_kwargs=False,
            ensure_available=True,
            model_version=model_version,
            runtime=runtime,
            **attrs,
        )
        os.environ["OPENLLM_MODEL_ID"] = str(llm.tag)

        labels = dict(llm.identifying_params)
        labels.update({"_type": llm.llm_type, "_framework": framework_envvar})
        workers_per_resource = first_not_none(workers_per_resource, default=llm_config["workers_per_resource"])

        with fs.open_fs(f"temp://llm_{llm_config['model_name']}") as llm_fs:
            dockerfile_template_path = None
            if dockerfile_template:
                with dockerfile_template:
                    llm_fs.writetext("Dockerfile.template", dockerfile_template.read())
                dockerfile_template_path = llm_fs.getsyspath("/Dockerfile.template")

            bento_tag = bentoml.Tag.from_taglike(f"{llm.llm_type}-service:{llm.tag.version}")
            try:
                bento = bentoml.get(bento_tag)
                if overwrite:
                    if output == "pretty":
                        _echo(f"Overwriting existing Bento {bento_tag}", fg="yellow")
                    bentoml.delete(bento_tag)
                    bento = create_bento(
                        bento_tag,
                        llm_fs,
                        llm,
                        workers_per_resource=workers_per_resource,
                        adapter_map=adapter_map,
                        quantize=quantize,
                        bettertransformer=bettertransformer,
                        extra_dependencies=enable_features,
                        build_ctx=build_ctx,
                        dockerfile_template=dockerfile_template_path,
                        runtime=runtime,
                    )
                _previously_built = True
            except bentoml.exceptions.NotFound:
                bento = create_bento(
                    bento_tag,
                    llm_fs,
                    llm,
                    workers_per_resource=workers_per_resource,
                    adapter_map=adapter_map,
                    quantize=quantize,
                    bettertransformer=bettertransformer,
                    extra_dependencies=enable_features,
                    build_ctx=build_ctx,
                    dockerfile_template=dockerfile_template_path,
                    runtime=runtime,
                )
    except Exception as e:
        logger.error("\nException caught during building LLM %s: \n", model_name, exc_info=e)
        raise
    else:
        del os.environ["OPENLLM_MODEL"]
        del os.environ["OPENLLM_MODEL_ID"]
        del os.environ["OPENLLM_ADAPTER_MAP"]
        # restore original OPENLLM_MODEL envvar if set.
        if current_model_envvar is not None:
            os.environ["OPENLLM_MODEL"] = current_model_envvar
        if current_model_id_envvar is not None:
            os.environ["OPENLLM_MODEL_ID"] = current_model_id_envvar
        if current_adapter_map_envvar is not None:
            os.environ["OPENLLM_ADAPTER_MAP"] = current_adapter_map_envvar

    if output == "pretty":
        if not get_quiet_mode():
            _echo("\n" + OPENLLM_FIGLET, fg="white")
            if not _previously_built:
                _echo(f"Successfully built {bento}.", fg="green")
            elif not overwrite:
                _echo(
                    f"'{model_name}' already has a Bento built [{bento}]. To overwrite it pass '--overwrite'.",
                    fg="yellow",
                )

            _echo(
                "📖 Next steps:\n\n"
                + "* Serving BentoLLM locally with 'openllm start':\n"
                + f"    $ openllm start {bento.tag}\n\n"
                + "* Push to BentoCloud with 'bentoml push':\n"
                + f"    $ bentoml push {bento.tag}\n\n"
                + "* Containerize your Bento with 'bentoml containerize':\n"
                + f"    $ bentoml containerize {bento.tag}"
                + " --opt progress=plain"
                if get_debug_mode()
                else ""
                + "\n\n"
                + "    Tip: To enable additional BentoML features for 'containerize', "
                + "use '--enable-features=FEATURE[,FEATURE]' "
                + "[see 'bentoml containerize -h' for more advanced usage]\n",
                fg="blue",
            )
    elif output == "json":
        _echo(orjson.dumps(bento.info.to_dict(), option=orjson.OPT_INDENT_2).decode())
    else:
        _echo(bento.tag)

    if format == "bento":
        return bento

    backend = os.getenv("BENTOML_CONTAINERIZE_BACKEND", "docker")
    _echo(f"\nBuilding {bento} into a LLMContainer using backend '{backend}'", fg="magenta")
    args = [str(bento.tag), "--backend", backend]
    if get_debug_mode():
        args.extend(["--opt", "progress=plain"])
    bentoml_cli.commands["containerize"].main(standalone_mode=False, args=args)
    return str(bento.tag)


@overload
def models_command(
    ctx: click.Context, output: OutputLiteral, show_available: bool, machine: t.Literal[True] = True
) -> DictStrAny:
    ...


@overload
def models_command(
    ctx: click.Context, output: OutputLiteral, show_available: bool, machine: t.Literal[False] = ...
) -> None:
    ...


@cli.command()
@output_option
@click.option(
    "--show-available",
    is_flag=True,
    default=False,
    help="Show available models in local store (mutually exclusive with '-o porcelain').",
)
@click.option("--machine", is_flag=True, default=False, hidden=True)
@click.pass_context
def models_command(
    ctx: click.Context, output: OutputLiteral, show_available: bool, machine: bool
) -> DictStrAny | None:
    """List all supported models.

    \b
    > NOTE: '--show-available' and '-o porcelain' are mutually exclusive.

    \b
    ```bash
    openllm models --show-available
    ```
    """
    from ._llm import normalise_model_name

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
            str, dict[t.Literal["model_id", "url", "installation", "cpu", "gpu", "runtime_impl"], t.Any]
        ] = {}

        converted: list[str] = []
        for m in models:
            config = openllm.AutoConfig.for_model(m)
            runtime_impl: tuple[str, ...] = ()
            if config["model_name"] in openllm.MODEL_MAPPING_NAMES:
                runtime_impl += ("pt",)
            if config["model_name"] in openllm.MODEL_FLAX_MAPPING_NAMES:
                runtime_impl += ("flax",)
            if config["model_name"] in openllm.MODEL_TF_MAPPING_NAMES:
                runtime_impl += ("tf",)
            json_data[m] = {
                "model_id": config["model_ids"],
                "url": config["url"],
                "cpu": not config["requires_gpu"],
                "gpu": True,
                "runtime_impl": runtime_impl,
                "installation": "pip install openllm"
                if m not in openllm.utils.OPTIONAL_DEPENDENCIES
                else f'pip install "openllm[{m}]"',
            }
            converted.extend([normalise_model_name(i) for i in config["model_ids"]])
            if DEBUG:
                try:
                    openllm.AutoLLM.for_model(m, llm_config=config)
                except Exception as err:
                    failed_initialized.append((m, err))

        ids_in_local_store = None
        if show_available:
            ids_in_local_store = {k: [i for i in bentoml.models.list() if k in i.tag.name] for k in json_data.keys()}
            ids_in_local_store = {k: v for k, v in ids_in_local_store.items() if v}

        if machine:
            dumped: DictStrAny = json_data
            if show_available:
                assert ids_in_local_store
                dumped["local"] = [bentoml_cattr.unstructure(i.tag) for m in ids_in_local_store.values() for i in m]
            return dumped
        elif output == "pretty":
            import tabulate

            tabulate.PRESERVE_WHITESPACE = True

            # llm, url, model_id, installation, cpu, gpu, runtime_impl
            data: list[
                str
                | tuple[
                    str,
                    str,
                    list[str],
                    str,
                    t.LiteralString,
                    t.LiteralString,
                    tuple[LiteralRuntime, ...],
                ]
            ] = []
            for m, v in json_data.items():
                data.extend(
                    [
                        (
                            m,
                            v["url"],
                            v["model_id"],
                            v["installation"],
                            "❌" if not v["cpu"] else "✅",
                            "✅",
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
                headers=["LLM", "URL", "Models Id", "Installation", "CPU", "GPU", "Runtime"],
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
                assert ids_in_local_store is not None
                if len(ids_in_local_store) == 0:
                    _echo("No models available locally.")
                    ctx.exit(0)

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
            dumped: DictStrAny = json_data
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
    ctx.exit(0)


@cli.command()
@click.option(
    "-y",
    "--yes",
    "--assume-yes",
    is_flag=True,
    help="Skip confirmation when deleting a specific model",
)
@inject
def prune_command(yes: bool, model_store: ModelStore = Provide[BentoMLContainer.model_store]):
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


def parsing_instruction_callback(
    ctx: click.Context, param: click.Parameter, value: list[str] | str | None
) -> tuple[str, bool | str] | list[str] | str | None:
    if value is None:
        return value

    if isinstance(value, list):
        # we only parse --text foo bar -> --text foo and omit bar
        value = value[-1]

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
    _memoized: DictStrAny,
    **attrs: t.Any,
):
    """Instruct agents interactively for given tasks, from a terminal.

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

    input_fg = "yellow"
    generated_fg = "cyan"

    if output != "porcelain":
        _echo("Input prompt: ", nl=False, fg="white")
        _echo(f"{prompt}", fg="magenta", nl=False)

    res = client.query(prompt, return_raw_response=True)

    if output == "pretty":
        formatted = client.llm.postprocess_generate(prompt, res["responses"])
        _echo("\n\n==Responses==\n", fg="white")
        _echo(f"{prompt} ", fg=input_fg, nl=False)
        _echo(formatted, fg=generated_fg)
    elif output == "json":
        _echo(orjson.dumps(res, option=orjson.OPT_INDENT_2).decode(), fg="white")
    else:
        _echo(res["responses"], fg="white")


def load_notebook_metadata() -> DictStrAny:
    with open(os.path.join(os.path.dirname(openllm.playground.__file__), "_meta.yml"), "r") as f:
        content = yaml.safe_load(f)
    if not all("description" in k for k in content.values()):
        raise ValueError("Invalid metadata file. All entries must have a 'description' key.")
    return content


@cli.command()
@click.argument("output-dir", default=".")
@click.option(
    "--port",
    envvar="JUPYTER_PORT",
    show_envvar=True,
    show_default=True,
    default=8888,
    help="Default port for Jupyter server",
)
def playground(output_dir: str | None, port: int):
    """OpenLLM Playground.

    A collections of notebooks to explore the capabilities of OpenLLM.
    This includes notebooks for fine-tuning, inference, and more.

    All of the script available in the playground can also be run directly as a Python script:
    For example:

    \b
    ```bash
    python -m openllm.playground.falcon_tuned --help
    ```

    \b
    > Note: This command requires Jupyter to be installed. Install it with 'pip install "openllm[playground]"'
    """
    if not is_jupyter_available() or not is_jupytext_available() or not is_notebook_available():
        raise RuntimeError(
            "Playground requires 'jupyter', 'jupytext', and 'notebook'. Install it with 'pip install \"openllm[playground]\"'"
        )

    import jupytext
    import nbformat

    metadata = load_notebook_metadata()
    _temp_dir = False
    if output_dir is None:
        _temp_dir = True
        output_dir = tempfile.mkdtemp(prefix="openllm-playground-")
    else:
        os.makedirs(os.path.abspath(os.path.expandvars(os.path.expanduser(output_dir))), exist_ok=True)

    _echo("The playground notebooks will be saved to: " + os.path.abspath(output_dir), fg="blue")
    for module in pkgutil.iter_modules(openllm.playground.__path__):
        if module.ispkg or os.path.exists(os.path.join(output_dir, module.name + ".ipynb")):
            logger.debug(
                "Skipping: %s (%s)",
                module.name,
                "File already exists" if not module.ispkg else f"{module.name} is a module",
            )
            continue
        _echo("Generating notebook for: " + module.name, fg="magenta")
        markdown_cell = nbformat.v4.new_markdown_cell(metadata[module.name]["description"])
        f = jupytext.read(os.path.join(module.module_finder.path, module.name + ".py"))  # type: ignore
        f.cells.insert(0, markdown_cell)
        jupytext.write(f, os.path.join(output_dir, module.name + ".ipynb"), fmt="notebook")
    try:
        subprocess.check_output(
            [
                sys.executable,
                "-m",
                "jupyter",
                "notebook",
                "--notebook-dir",
                output_dir,
                "--port",
                str(port),
                "--no-browser",
                "--debug",
            ]
        )
    except subprocess.CalledProcessError as e:
        _echo(e.output, fg="red")
        raise e
    except KeyboardInterrupt:
        _echo("\nShutting down Jupyter server...", fg="yellow")
        if _temp_dir:
            _echo("Note: You can access the generated notebooks in: " + output_dir, fg="blue")


if psutil.WINDOWS:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore


if __name__ == "__main__":
    cli()
