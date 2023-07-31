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

from __future__ import annotations
import importlib.util
import os
import typing as t

import click
import click_option_group as cog
import inflection
import orjson
from bentoml_cli.utils import BentoMLCommandGroup
from click.shell_completion import CompletionItem

import bentoml
from bentoml._internal.configuration.containers import BentoMLContainer

from . import termui
from .. import bundle
from ..models.auto import CONFIG_MAPPING
from ..models.auto import AutoConfig
from ..utils import ENV_VARS_TRUE_VALUES
from ..utils import EnvVarMixin
from ..utils import LazyType
from ..utils import analytics
from ..utils import available_devices
from ..utils import compose
from ..utils import dantic
from ..utils import device_count
from ..utils import first_not_none
from ..utils import get_debug_mode
from ..utils import get_quiet_mode
from ..utils import infer_auto_class
from ..utils import is_peft_available
from ..utils import resolve_user_filepath

if t.TYPE_CHECKING:
  import subprocess

  from .._configuration import LLMConfig
  from .._types import DictStrAny
  from .._types import P
  TupleStr = tuple[str, ...]
else:
  TupleStr = tuple

LiteralOutput = t.Literal["json", "pretty", "porcelain"]

_AnyCallable = t.Callable[..., t.Any]
FC = t.TypeVar("FC", bound=t.Union[_AnyCallable, click.Command])

def parse_config_options(config: LLMConfig, server_timeout: int, workers_per_resource: float, device: tuple[str, ...] | None, environ: DictStrAny,) -> DictStrAny:
  # TODO: Support amd.com/gpu on k8s
  _bentoml_config_options_env = environ.pop("BENTOML_CONFIG_OPTIONS", "")
  _bentoml_config_options_opts = [
      "tracing.sample_rate=1.0", f"api_server.traffic.timeout={server_timeout}", f'runners."llm-{config["start_name"]}-runner".traffic.timeout={config["timeout"]}', f'runners."llm-{config["start_name"]}-runner".workers_per_resource={workers_per_resource}'
  ]
  if device:
    if len(device) > 1: _bentoml_config_options_opts.extend([f'runners."llm-{config["start_name"]}-runner".resources."nvidia.com/gpu"[{idx}]={dev}' for idx, dev in enumerate(device)])
    else: _bentoml_config_options_opts.append(f'runners."llm-{config["start_name"]}-runner".resources."nvidia.com/gpu"=[{device[0]}]')
  _bentoml_config_options_env += " " if _bentoml_config_options_env else "" + " ".join(_bentoml_config_options_opts)
  environ["BENTOML_CONFIG_OPTIONS"] = _bentoml_config_options_env
  return environ

_adapter_mapping_key = "adapter_map"

def _id_callback(ctx: click.Context, _: click.Parameter, value: tuple[str, ...] | None) -> None:
  if not value: return None
  if _adapter_mapping_key not in ctx.params: ctx.params[_adapter_mapping_key] = {}
  for v in value:
    adapter_id, *adapter_name = v.rsplit(":", maxsplit=1)
    # try to resolve the full path if users pass in relative,
    # currently only support one level of resolve path with current directory
    try:
      adapter_id = resolve_user_filepath(adapter_id, os.getcwd())
    except FileNotFoundError:
      pass
    ctx.params[_adapter_mapping_key][adapter_id] = adapter_name[0] if len(adapter_name) > 0 else None
  return None

def start_command_factory(group: click.Group, model: str, _context_settings: DictStrAny | None = None, _serve_grpc: bool = False) -> click.Command:
  """Generate a 'click.Command' for any given LLM.

  Args:
  group: the target ``click.Group`` to save this LLM cli under
  model: The name of the model or the ``bentoml.Bento`` instance.

  Returns:
  The click.Command for starting the model server

  Note that the internal commands will return the llm_config and a boolean determine
  whether the server is run with GPU or not.
  """
  llm_config = AutoConfig.for_model(model)

  command_attrs: DictStrAny = dict(
      name=llm_config["model_name"],
      context_settings=_context_settings or termui.CONTEXT_SETTINGS,
      short_help=f"Start a LLMServer for '{model}'",
      aliases=[llm_config["start_name"]] if llm_config["name_type"] == "dasherize" else None,
      help=f"""\
{llm_config['env'].start_docstring}

\b
Note: ``{llm_config['start_name']}`` can also be run with any other models available on HuggingFace
or fine-tuned variants as long as it belongs to the architecture generation ``{llm_config['architecture']}`` (trust_remote_code={llm_config['trust_remote_code']}).

\b
For example: One can start [Fastchat-T5](https://huggingface.co/lmsys/fastchat-t5-3b-v1.0) with ``openllm start flan-t5``:

\b
$ openllm start flan-t5 --model-id lmsys/fastchat-t5-3b-v1.0

\b
Available official model_id(s): [default: {llm_config['default_id']}]

\b
{orjson.dumps(llm_config['model_ids'], option=orjson.OPT_INDENT_2).decode()}
""",
  )

  if llm_config["requires_gpu"] and device_count() < 1:
    # NOTE: The model requires GPU, therefore we will return a dummy command
    command_attrs.update({"short_help": "(Disabled because there is no GPU available)", "help": f"""{model} is currently not available to run on your local machine because it requires GPU for inference."""})
    return noop_command(group, llm_config, _serve_grpc, **command_attrs)

  @group.command(**command_attrs)
  @start_decorator(llm_config, serve_grpc=_serve_grpc)
  @click.pass_context
  def start_cmd(
      ctx: click.Context, /, server_timeout: int, model_id: str | None, model_version: str | None, workers_per_resource: t.Literal["conserved", "round_robin"] | t.LiteralString, device: tuple[str, ...], quantize: t.Literal["int8", "int4", "gptq"] | None,
      bettertransformer: bool | None, runtime: t.Literal["ggml", "transformers"], fast: bool, serialisation_format: t.Literal["safetensors", "legacy"], adapter_id: str | None, return_process: bool, **attrs: t.Any,
  ) -> LLMConfig | subprocess.Popen[bytes]:
    fast = str(fast).upper() in ENV_VARS_TRUE_VALUES
    if serialisation_format == "safetensors" and quantize is not None and os.getenv("OPENLLM_SERIALIZATION_WARNING", str(True)).upper() in ENV_VARS_TRUE_VALUES:
      termui.echo(
          f"'--quantize={quantize}' might not work with 'safetensors' serialisation format. Use with caution!. To silence this warning, set \"OPENLLM_SERIALIZATION_WARNING=False\"\nNote: You can always fallback to '--serialisation legacy' when running quantisation.",
          fg="yellow"
      )
    adapter_map: dict[str, str | None] | None = attrs.pop(_adapter_mapping_key, None)
    config, server_attrs = llm_config.model_validate_click(**attrs)
    server_timeout = first_not_none(server_timeout, default=config["timeout"])
    server_attrs.update({"working_dir": os.path.dirname(os.path.dirname(__file__)), "timeout": server_timeout})
    if _serve_grpc: server_attrs["grpc_protocol_version"] = "v1"
    # NOTE: currently, theres no development args in bentoml.Server. To be fixed upstream.
    development = server_attrs.pop("development")
    server_attrs.setdefault("production", not development)
    wpr = first_not_none(workers_per_resource, default=config["workers_per_resource"])

    if isinstance(wpr, str):
      if wpr == "round_robin": wpr = 1.0
      elif wpr == "conserved":
        if device and device_count() == 0:
          termui.echo("--device will have no effect as there is no GPUs available", fg="yellow")
          wpr = 1.0
        else:
          available_gpu = len(device) if device else device_count()
          wpr = 1.0 if available_gpu == 0 else float(1 / available_gpu)
      else:
        wpr = float(wpr)
    elif isinstance(wpr, int):
      wpr = float(wpr)

    # Create a new model env to work with the envvar during CLI invocation
    env = EnvVarMixin(config["model_name"], config.default_implementation(), model_id=model_id, bettertransformer=bettertransformer, quantize=quantize, runtime=runtime)
    prerequisite_check(ctx, config, quantize, adapter_map, int(1 / wpr))

    # NOTE: This is to set current configuration
    start_env = os.environ.copy()
    start_env = parse_config_options(config, server_timeout, wpr, device, start_env)
    if fast: termui.echo(f"Fast mode is enabled. Make sure the model is available in local store before 'start': 'openllm import {model}{' --model-id ' + model_id if model_id else ''}'", fg="yellow")

    start_env.update({
        "OPENLLM_MODEL": model,
        "BENTOML_DEBUG": str(get_debug_mode()),
        "BENTOML_HOME": os.getenv("BENTOML_HOME", BentoMLContainer.bentoml_home.get()),
        "OPENLLM_ADAPTER_MAP": orjson.dumps(adapter_map).decode(),
        "OPENLLM_SERIALIZATION": serialisation_format,
        env.runtime: env.runtime_value,
        env.framework: env.framework_value,
    })
    if env.model_id_value: start_env[env.model_id] = str(env.model_id_value)
    # NOTE: quantize and bettertransformer value is already assigned within env
    if bettertransformer is not None: start_env[env.bettertransformer] = str(env.bettertransformer_value)
    if quantize is not None: start_env[env.quantize] = str(env.quantize_value)

    llm = infer_auto_class(env.framework_value).for_model(model, model_version=model_version, llm_config=config, ensure_available=not fast, adapter_map=adapter_map, serialisation=serialisation_format)
    start_env.update({env.config: llm.config.model_dump_json().decode(), env.model_id: llm.model_id})

    server = bentoml.GrpcServer("_service.py:svc", **server_attrs) if _serve_grpc else bentoml.HTTPServer("_service.py:svc", **server_attrs)
    analytics.track_start_init(llm.config)

    def next_step(model_name: str, adapter_map: DictStrAny | None) -> None:
      cmd_name = f"openllm build {model_name}"
      if adapter_map is not None: cmd_name += " " + " ".join([f"--adapter-id {s}" for s in [f"{p}:{name}" if name not in (None, "default") else p for p, name in adapter_map.items()]])
      if not get_quiet_mode(): termui.echo(f"\n🚀 Next step: run '{cmd_name}' to create a Bento for {model_name}", fg="blue")

    if return_process:
      server.start(env=start_env, text=True)
      if server.process is None: raise click.ClickException("Failed to start the server.")
      return server.process
    else:
      try:
        server.start(env=start_env, text=True, blocking=True)
      except KeyboardInterrupt:
        next_step(model, adapter_map)
      except Exception as err:
        termui.echo(f"Error caught while running LLM Server:\n{err}", fg="red")
      else:
        next_step(model, adapter_map)

    # NOTE: Return the configuration for telemetry purposes.
    return config

  return start_cmd

def noop_command(group: click.Group, llm_config: LLMConfig, _serve_grpc: bool, **command_attrs: t.Any) -> click.Command:
  context_settings = command_attrs.pop("context_settings", {})
  context_settings.update({"ignore_unknown_options": True, "allow_extra_args": True})
  command_attrs["context_settings"] = context_settings
  # NOTE: The model requires GPU, therefore we will return a dummy command
  @group.command(**command_attrs)
  def noop(**_: t.Any) -> LLMConfig:
    termui.echo("No GPU available, therefore this command is disabled", fg="red")
    analytics.track_start_init(llm_config)
    return llm_config

  return noop

def prerequisite_check(ctx: click.Context, llm_config: LLMConfig, quantize: t.LiteralString | None, adapter_map: dict[str, str | None] | None, num_workers: int) -> None:
  if adapter_map and not is_peft_available(): ctx.fail("Using adapter requires 'peft' to be available. Make sure to install with 'pip install \"openllm[fine-tune]\"'")
  if quantize and llm_config.default_implementation() == "vllm": ctx.fail(f"Quantization is not yet supported with vLLM. Set '{llm_config.env['framework']}=\"pt\"' to run with quantization.")
  requirements = llm_config["requirements"]
  if requirements is not None and len(requirements) > 0:
    missing_requirements = [i for i in requirements if importlib.util.find_spec(inflection.underscore(i)) is None]
    if len(missing_requirements) > 0: termui.echo(f"Make sure to have the following dependencies available: {missing_requirements}", fg="yellow")

def start_decorator(llm_config: LLMConfig, serve_grpc: bool = False) -> t.Callable[[FC], t.Callable[[FC], FC]]:
  return lambda fn: compose(
      *[
          llm_config.to_click_options, _http_server_args if not serve_grpc else _grpc_server_args,
          cog.optgroup.group("General LLM Options", help=f"The following options are related to running '{llm_config['start_name']}' LLM Server."),
          model_id_option(factory=cog.optgroup, model_env=llm_config["env"]),
          model_version_option(factory=cog.optgroup),
          cog.optgroup.option("--server-timeout", type=int, default=None, help="Server timeout in seconds"),
          workers_per_resource_option(factory=cog.optgroup),
          fast_option(factory=cog.optgroup),
          cog.optgroup.group(
              "LLM Optimization Options",
              help="""Optimization related options.

            OpenLLM supports running model with [BetterTransformer](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/),
            k-bit quantization (8-bit, 4-bit), GPTQ quantization, PagedAttention via vLLM.

            The following are either in our roadmap or currently being worked on:

            - DeepSpeed Inference: [link](https://www.deepspeed.ai/inference/)
            - GGML: Fast inference on [bare metal](https://github.com/ggerganov/ggml)
            """,
          ),
          cog.optgroup.option("--device", type=dantic.CUDA, multiple=True, envvar="CUDA_VISIBLE_DEVICES", callback=parse_device_callback, help=f"Assign GPU devices (if available) for {llm_config['model_name']}.", show_envvar=True),
          cog.optgroup.option("--runtime", type=click.Choice(["ggml", "transformers"]), default="transformers", help="The runtime to use for the given model. Default is transformers."),
          quantize_option(factory=cog.optgroup, model_env=llm_config["env"]),
          bettertransformer_option(factory=cog.optgroup, model_env=llm_config["env"]),
          serialisation_option(factory=cog.optgroup),
          cog.optgroup.group(
              "Fine-tuning related options",
              help="""\
    Note that the argument `--adapter-id` can accept the following format:

    - `--adapter-id /path/to/adapter` (local adapter)

    - `--adapter-id remote/adapter` (remote adapter from HuggingFace Hub)

    - `--adapter-id remote/adapter:eng_lora` (two previous adapter options with the given adapter_name)

    ```bash

    $ openllm start opt --adapter-id /path/to/adapter_dir --adapter-id remote/adapter:eng_lora

    ```
    """,
          ),
          cog.optgroup.option("--adapter-id", default=None, help="Optional name or path for given LoRA adapter" + f" to wrap '{llm_config['model_name']}'", multiple=True, callback=_id_callback, metavar="[PATH | [remote/][adapter_name:]adapter_id][, ...]"),
          click.option("--return-process", is_flag=True, default=False, help="Internal use only.", hidden=True),
      ]
  )(fn)

def parse_device_callback(ctx: click.Context, param: click.Parameter, value: tuple[tuple[str], ...] | None) -> TupleStr | None:
  if value is None: return value
  if not LazyType(TupleStr).isinstance(value): ctx.fail(f"{param} only accept multiple values, not {type(value)} (value: {value})")
  el: TupleStr = tuple(i for k in value for i in k)
  # NOTE: --device all is a special case
  if len(el) == 1 and el[0] == "all": return tuple(map(str, available_devices()))
  return el

# NOTE: A list of bentoml option that is not needed for parsing.
# NOTE: User shouldn't set '--working-dir', as OpenLLM will setup this.
# NOTE: production is also deprecated
_IGNORED_OPTIONS = {"working_dir", "production", "protocol_version"}

def parse_serve_args(serve_grpc: bool) -> t.Callable[[t.Callable[..., LLMConfig]], t.Callable[[FC], FC]]:
  """Parsing `bentoml serve|serve-grpc` click.Option to be parsed via `openllm start`."""
  from bentoml_cli.cli import cli

  command = "serve" if not serve_grpc else "serve-grpc"
  group = cog.optgroup.group(f"Start a {'HTTP' if not serve_grpc else 'gRPC'} server options", help=f"Related to serving the model [synonymous to `bentoml {'serve-http' if not serve_grpc else command }`]",)

  def decorator(f: t.Callable[t.Concatenate[int, str | None, P], LLMConfig]) -> t.Callable[[FC], FC]:
    serve_command = cli.commands[command]
    # The first variable is the argument bento
    # The last five is from BentoMLCommandGroup.NUMBER_OF_COMMON_PARAMS
    serve_options = [p for p in serve_command.params[1:-BentoMLCommandGroup.NUMBER_OF_COMMON_PARAMS] if p.name not in _IGNORED_OPTIONS]
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

_http_server_args, _grpc_server_args = parse_serve_args(False), parse_serve_args(True)

def cli_option(*param_decls: t.Any, **attrs: t.Any) -> t.Callable[[FC | None], FC]:
  """General ``@click.option`` with some sauce.

  This decorator extends the default ``@click.option`` plus a factory option to use which type of option, for example: [click, click_option_group.optgroup]
  """
  attrs.setdefault("help", "General option for OpenLLM CLI.")
  factory = attrs.pop("factory", click)

  def decorator(f: FC | None) -> FC:
    return t.cast(FC, factory.option(*param_decls, **attrs)(f) if f is not None else factory.option(*param_decls, **attrs))

  return decorator

def output_option(f: _AnyCallable | None = None, *, default_value: LiteralOutput = "pretty", **attrs: t.Any) -> t.Callable[[FC], FC]:
  output = ["json", "pretty", "porcelain"]

  def complete_output_var(ctx: click.Context, param: click.Parameter, incomplete: str) -> list[CompletionItem]:
    return [CompletionItem(it) for it in output]

  return cli_option("-o", "--output", "output", type=click.Choice(output), default=default_value, help="Showing output type.", show_default=True, envvar="OPENLLM_OUTPUT", show_envvar=True, shell_complete=complete_output_var, **attrs)(f)

def fast_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
      "--fast/--no-fast",
      show_default=True,
      default=False,
      envvar="OPENLLM_USE_LOCAL_LATEST",
      show_envvar=True,
      help="""Whether to skip checking if models is already in store.

                                                                                                          This is useful if you already downloaded or setup the model beforehand.
                                                                                                          """,
      **attrs
  )(f)

def machine_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option("--machine", is_flag=True, default=False, hidden=True, **attrs)(f)

def model_id_option(f: _AnyCallable | None = None, *, model_env: EnvVarMixin | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option("--model-id", type=click.STRING, default=None, envvar=model_env.model_id if model_env is not None else None, show_envvar=model_env is not None, help="Optional model_id name or path for (fine-tune) weight.", **attrs)(f)

def model_version_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option("--model-version", type=click.STRING, default=None, help="Optional model version to save for this model. It will be inferred automatically from model-id.", **attrs)(f)

def model_name_argument(f: _AnyCallable | None = None, required: bool = True) -> t.Callable[[FC], FC]:
  arg = click.argument("model_name", type=click.Choice([inflection.dasherize(name) for name in CONFIG_MAPPING]), required=required)
  return arg(f) if f is not None else arg

def quantize_option(f: _AnyCallable | None = None, *, build: bool = False, model_env: EnvVarMixin | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
      "--quantise",
      "--quantize",
      "quantize",
      type=click.Choice(["int8", "int4", "gptq"]),
      default=None,
      envvar=model_env.quantize if model_env is not None else None,
      show_envvar=model_env is not None,
      help="""Dynamic quantization for running this LLM.

                                                                                                                                                                            The following quantization strategies are supported:

                                                                                                                                                                            - ``int8``: ``LLM.int8`` for [8-bit](https://arxiv.org/abs/2208.07339) quantization.

                                                                                                                                                                            - ``int4``: ``SpQR`` for [4-bit](https://arxiv.org/abs/2306.03078) quantization.

                                                                                                                                                                            - ``gptq``: ``GPTQ`` [quantization](https://arxiv.org/abs/2210.17323)

                                                                                                                                                                            **Note** that the model can also be served with quantized weights.
                                                                                                                                                                            """ + (
          """
                                                                                                                                                                            **Note** that this will set the mode for serving within deployment."""
          if build else ""
      ) + """
                                                                                                                                                                            **Note** that quantization are currently only available in *PyTorch* models.""",
      **attrs
  )(f)

def workers_per_resource_option(f: _AnyCallable | None = None, *, build: bool = False, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
      "--workers-per-resource",
      default=None,
      callback=workers_per_resource_callback,
      type=str,
      required=False,
      help="""Number of workers per resource assigned.

                                                                                                                                                  See https://docs.bentoml.org/en/latest/guides/scheduling.html#resource-scheduling-strategy
                                                                                                                                                  for more information. By default, this is set to 1.

                                                                                                                                                  **Note**: ``--workers-per-resource`` will also accept the following strategies:

                                                                                                                                                  - ``round_robin``: Similar behaviour when setting ``--workers-per-resource 1``. This is useful for smaller models.

                                                                                                                                                  - ``conserved``: This will determine the number of available GPU resources, and only assign one worker for the LLMRunner. For example, if ther are 4 GPUs available, then ``conserved`` is equivalent to ``--workers-per-resource 0.25``.
                                                                                                                                                  """ + (
          """\n
                                                                                                                                                  **Note**: The workers value passed into 'build' will determine how the LLM can
                                                                                                                                                  be provisioned in Kubernetes as well as in standalone container. This will
                                                                                                                                                  ensure it has the same effect with 'openllm start --workers ...'""" if build else ""
      ),
      **attrs
  )(f)

def bettertransformer_option(f: _AnyCallable | None = None, *, build: bool = False, model_env: EnvVarMixin | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
      "--bettertransformer",
      is_flag=True,
      default=None,
      envvar=model_env.bettertransformer if model_env is not None else None,
      show_envvar=model_env is not None,
      help="Apply FasterTransformer wrapper to serve model. This will applies during serving time." if not build else "Set default environment variable whether to serve this model with FasterTransformer in build time.",
      **attrs
  )(f)

def serialisation_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
      "--serialisation",
      "--serialization",
      "serialisation_format",
      type=click.Choice(["safetensors", "legacy"]),
      default="safetensors",
      show_default=True,
      show_envvar=True,
      envvar="OPENLLM_SERIALIZATION",
      help="""Serialisation format for save/load LLM.

                                                                                                                   Currently the following strategies are supported:

                                                                                                                   - ``safetensors``: This will use safetensors format, which is synonymous to

                                                                                                                               \b
                                                                                                                               ``safe_serialization=True``.

                                                                                                                               \b
                                                                                                                               **Note** that this format might not work for every cases, and
                                                                                                                               you can always fallback to ``legacy`` if needed.

                                                                                                                   - ``legacy``: This will use PyTorch serialisation format, often as ``.bin`` files.
                                                                                                                                   This should be used if the model doesn't yet support safetensors.

                                                                                                                   **Note** that GGML format is working in progress.
                                                                                                                   """,
      **attrs
  )(f)

def container_registry_option(f: _AnyCallable | None = None, **attrs: t.Any) -> t.Callable[[FC], FC]:
  return cli_option(
      "--container-registry",
      "container_registry",
      type=str,
      default="ecr",
      show_default=True,
      show_envvar=True,
      envvar="OPENLLM_CONTAINER_REGISTRY",
      callback=container_registry_callback,
      help="""The default container registry to get the base image for building BentoLLM.

                                                                                                                        Currently, it supports 'ecr', 'ghcr.io', 'docker.io'

                                                                                                                        \b
                                                                                                                        **Note** that in order to build the base image, you will need a GPUs to compile custom kernel. See ``openllm ext build-base-container`` for more information.
                                                                                                                        """
  )(f)

_wpr_strategies = {"round_robin", "conserved"}

def workers_per_resource_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
  if value is None: return value
  value = inflection.underscore(value)
  if value in _wpr_strategies: return value
  else:
    try:
      float(value)  # type: ignore[arg-type]
    except ValueError:
      raise click.BadParameter(f"'workers_per_resource' only accept '{_wpr_strategies}' as possible strategies, otherwise pass in float.", ctx, param) from None
    else:
      return value

def container_registry_callback(ctx: click.Context, param: click.Parameter, value: str | None) -> str | None:
  if value is None: return value
  if value not in bundle.supported_registries: raise click.BadParameter(f"Value must be one of {bundle.supported_registries}", ctx, param)
  return value
