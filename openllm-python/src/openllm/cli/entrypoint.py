"""OpenLLM CLI interface.

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
import http.client
import inspect
import itertools
import logging
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import typing as t

import attr
import click
import click_option_group as cog
import fs
import fs.copy
import fs.errors
import inflection
import orjson

from bentoml_cli.utils import BentoMLCommandGroup
from bentoml_cli.utils import opt_callback
from simple_di import Provide
from simple_di import inject

import bentoml
import openllm

from bentoml._internal.configuration.containers import BentoMLContainer
from bentoml._internal.models.model import ModelStore
from openllm import bundle
from openllm.exceptions import OpenLLMException
from openllm.models.auto import CONFIG_MAPPING
from openllm.models.auto import MODEL_FLAX_MAPPING_NAMES
from openllm.models.auto import MODEL_MAPPING_NAMES
from openllm.models.auto import MODEL_TF_MAPPING_NAMES
from openllm.models.auto import MODEL_VLLM_MAPPING_NAMES
from openllm.models.auto import AutoConfig
from openllm.models.auto import AutoLLM
from openllm.utils import infer_auto_class
from openllm_core._typing_compat import Concatenate
from openllm_core._typing_compat import DictStrAny
from openllm_core._typing_compat import LiteralBackend
from openllm_core._typing_compat import LiteralQuantise
from openllm_core._typing_compat import LiteralString
from openllm_core._typing_compat import ParamSpec
from openllm_core._typing_compat import Self
from openllm_core.utils import DEBUG
from openllm_core.utils import DEBUG_ENV_VAR
from openllm_core.utils import OPTIONAL_DEPENDENCIES
from openllm_core.utils import QUIET_ENV_VAR
from openllm_core.utils import EnvVarMixin
from openllm_core.utils import LazyLoader
from openllm_core.utils import analytics
from openllm_core.utils import bentoml_cattr
from openllm_core.utils import compose
from openllm_core.utils import configure_logging
from openllm_core.utils import first_not_none
from openllm_core.utils import get_debug_mode
from openllm_core.utils import get_quiet_mode
from openllm_core.utils import is_torch_available
from openllm_core.utils import resolve_user_filepath
from openllm_core.utils import set_debug_mode
from openllm_core.utils import set_quiet_mode

from . import termui
from ._factory import FC
from ._factory import LiteralOutput
from ._factory import _AnyCallable
from ._factory import backend_option
from ._factory import container_registry_option
from ._factory import machine_option
from ._factory import model_id_option
from ._factory import model_name_argument
from ._factory import model_version_option
from ._factory import output_option
from ._factory import quantize_option
from ._factory import serialisation_option
from ._factory import start_command_factory
from ._factory import workers_per_resource_option

if t.TYPE_CHECKING:
  import torch

  from bentoml._internal.bento import BentoStore
  from bentoml._internal.container import DefaultBuilder
  from openllm_core._schema import EmbeddingsOutput
  from openllm_core._typing_compat import LiteralContainerRegistry
  from openllm_core._typing_compat import LiteralContainerVersionStrategy
else:
  torch = LazyLoader('torch', globals(), 'torch')

P = ParamSpec('P')
logger = logging.getLogger(__name__)
OPENLLM_FIGLET = '''\
 ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝
'''

ServeCommand = t.Literal['serve', 'serve-grpc']
@attr.define
class GlobalOptions:
  cloud_context: str | None = attr.field(default=None)

  def with_options(self, **attrs: t.Any) -> Self:
    return attr.evolve(self, **attrs)
GrpType = t.TypeVar('GrpType', bound=click.Group)

_object_setattr = object.__setattr__

_EXT_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), 'extension'))
class Extensions(click.MultiCommand):
  def list_commands(self, ctx: click.Context) -> list[str]:
    return sorted([filename[:-3] for filename in os.listdir(_EXT_FOLDER) if filename.endswith('.py') and not filename.startswith('__')])

  def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
    try:
      mod = __import__(f'openllm.cli.extension.{cmd_name}', None, None, ['cli'])
    except ImportError:
      return None
    return mod.cli
class OpenLLMCommandGroup(BentoMLCommandGroup):
  NUMBER_OF_COMMON_PARAMS = 5  # parameters in common_params + 1 faked group option header

  @staticmethod
  def common_params(f: t.Callable[P, t.Any]) -> t.Callable[[FC], FC]:
    # The following logics is similar to one of BentoMLCommandGroup
    @cog.optgroup.group(name='Global options', help='Shared globals options for all OpenLLM CLI.')
    @cog.optgroup.option('-q', '--quiet', envvar=QUIET_ENV_VAR, is_flag=True, default=False, help='Suppress all output.', show_envvar=True)
    @cog.optgroup.option('--debug', '--verbose', 'debug', envvar=DEBUG_ENV_VAR, is_flag=True, default=False, help='Print out debug logs.', show_envvar=True)
    @cog.optgroup.option('--do-not-track', is_flag=True, default=False, envvar=analytics.OPENLLM_DO_NOT_TRACK, help='Do not send usage info', show_envvar=True)
    @cog.optgroup.option('--context', 'cloud_context', envvar='BENTOCLOUD_CONTEXT', type=click.STRING, default=None, help='BentoCloud context name.', show_envvar=True)
    @click.pass_context
    @functools.wraps(f)
    def wrapper(ctx: click.Context, quiet: bool, debug: bool, cloud_context: str | None, *args: P.args, **attrs: P.kwargs) -> t.Any:
      ctx.obj = GlobalOptions(cloud_context=cloud_context)
      if quiet:
        set_quiet_mode(True)
        if debug: logger.warning("'--quiet' passed; ignoring '--verbose/--debug'")
      elif debug: set_debug_mode(True)
      configure_logging()
      return f(*args, **attrs)

    return wrapper

  @staticmethod
  def usage_tracking(func: t.Callable[P, t.Any], group: click.Group, **attrs: t.Any) -> t.Callable[Concatenate[bool, P], t.Any]:
    command_name = attrs.get('name', func.__name__)

    @functools.wraps(func)
    def wrapper(do_not_track: bool, *args: P.args, **attrs: P.kwargs) -> t.Any:
      if do_not_track:
        with analytics.set_bentoml_tracking():
          return func(*args, **attrs)
      start_time = time.time_ns()
      with analytics.set_bentoml_tracking():
        if group.name is None: raise ValueError('group.name should not be None')
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

    return t.cast(t.Callable[Concatenate[bool, P], t.Any], wrapper)

  def get_command(self, ctx: click.Context, cmd_name: str) -> click.Command | None:
    if cmd_name in t.cast('Extensions', extension_command).list_commands(ctx):
      return t.cast('Extensions', extension_command).get_command(ctx, cmd_name)
    cmd_name = self.resolve_alias(cmd_name)
    if ctx.command.name in _start_mapping:
      try:
        return _start_mapping[ctx.command.name][cmd_name]
      except KeyError:
        # TODO: support start from a bento
        try:
          bentoml.get(cmd_name)
          raise click.ClickException(f"'openllm start {cmd_name}' is currently disabled for the time being. Please let us know if you need this feature by opening an issue on GitHub.")
        except bentoml.exceptions.NotFound:
          pass
        raise click.BadArgumentUsage(f'{cmd_name} is not a valid model identifier supported by OpenLLM.') from None
    return super().get_command(ctx, cmd_name)

  def list_commands(self, ctx: click.Context) -> list[str]:
    if ctx.command.name in {'start', 'start-grpc'}: return list(CONFIG_MAPPING.keys())
    return super().list_commands(ctx) + t.cast('Extensions', extension_command).list_commands(ctx)

  def command(self, *args: t.Any, **kwargs: t.Any) -> t.Callable[[t.Callable[..., t.Any]], click.Command]:  # type: ignore[override] # XXX: fix decorator on BentoMLCommandGroup
    """Override the default 'cli.command' with supports for aliases for given command, and it wraps the implementation with common parameters."""
    if 'context_settings' not in kwargs: kwargs['context_settings'] = {}
    if 'max_content_width' not in kwargs['context_settings']: kwargs['context_settings']['max_content_width'] = 120
    aliases = kwargs.pop('aliases', None)

    def decorator(f: _AnyCallable) -> click.Command:
      name = f.__name__.lower()
      if name.endswith('_command'): name = name[:-8]
      name = name.replace('_', '-')
      kwargs.setdefault('help', inspect.getdoc(f))
      kwargs.setdefault('name', name)
      wrapped = self.usage_tracking(self.common_params(f), self, **kwargs)

      # move common parameters to end of the parameters list
      _memo = getattr(wrapped, '__click_params__', None)
      if _memo is None: raise ValueError('Click command not register correctly.')
      _object_setattr(wrapped, '__click_params__', _memo[-self.NUMBER_OF_COMMON_PARAMS:] + _memo[:-self.NUMBER_OF_COMMON_PARAMS])
      # NOTE: we need to call super of super to avoid conflict with BentoMLCommandGroup command setup
      cmd = super(BentoMLCommandGroup, self).command(*args, **kwargs)(wrapped)
      # NOTE: add aliases to a given commands if it is specified.
      if aliases is not None:
        if not cmd.name: raise ValueError('name is required when aliases are available.')
        self._commands[cmd.name] = aliases
        self._aliases.update({alias: cmd.name for alias in aliases})
      return cmd

    return decorator

  def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
    '''Additional format methods that include extensions as well as the default cli command.'''
    from gettext import gettext as _
    commands: list[tuple[str, click.Command]] = []
    extensions: list[tuple[str, click.Command]] = []
    _cached_extensions: list[str] = t.cast('Extensions', extension_command).list_commands(ctx)
    for subcommand in self.list_commands(ctx):
      cmd = self.get_command(ctx, subcommand)
      if cmd is None or cmd.hidden: continue
      if subcommand in _cached_extensions: extensions.append((subcommand, cmd))
      else: commands.append((subcommand, cmd))
    # allow for 3 times the default spacing
    if len(commands):
      limit = formatter.width - 6 - max(len(cmd[0]) for cmd in commands)
      rows: list[tuple[str, str]] = []
      for subcommand, cmd in commands:
        help = cmd.get_short_help_str(limit)
        rows.append((subcommand, help))
      if rows:
        with formatter.section(_('Commands')):
          formatter.write_dl(rows)
    if len(extensions):
      limit = formatter.width - 6 - max(len(cmd[0]) for cmd in extensions)
      rows = []
      for subcommand, cmd in extensions:
        help = cmd.get_short_help_str(limit)
        rows.append((inflection.dasherize(subcommand), help))
      if rows:
        with formatter.section(_('Extensions')):
          formatter.write_dl(rows)
@click.group(cls=OpenLLMCommandGroup, context_settings=termui.CONTEXT_SETTINGS, name='openllm')
@click.version_option(
    None, '--version', '-v', message=f"%(prog)s, %(version)s (compiled: {'yes' if openllm.COMPILED else 'no'})\nPython ({platform.python_implementation()}) {platform.python_version()}"
)
def cli() -> None:
  '''\b
   ██████╗ ██████╗ ███████╗███╗   ██╗██╗     ██╗     ███╗   ███╗
  ██╔═══██╗██╔══██╗██╔════╝████╗  ██║██║     ██║     ████╗ ████║
  ██║   ██║██████╔╝█████╗  ██╔██╗ ██║██║     ██║     ██╔████╔██║
  ██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║██║     ██║     ██║╚██╔╝██║
  ╚██████╔╝██║     ███████╗██║ ╚████║███████╗███████╗██║ ╚═╝ ██║
   ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝     ╚═╝.

  \b
  An open platform for operating large language models in production.
  Fine-tune, serve, deploy, and monitor any LLMs with ease.
  '''
@cli.group(cls=OpenLLMCommandGroup, context_settings=termui.CONTEXT_SETTINGS, name='start', aliases=['start-http'])
def start_command() -> None:
  '''Start any LLM as a REST server.

  \b
  ```bash
  $ openllm <start|start-http> <model_name> --<options> ...
  ```
  '''
@cli.group(cls=OpenLLMCommandGroup, context_settings=termui.CONTEXT_SETTINGS, name='start-grpc')
def start_grpc_command() -> None:
  '''Start any LLM as a gRPC server.

  \b
  ```bash
  $ openllm start-grpc <model_name> --<options> ...
  ```
  '''
_start_mapping = {
    'start': {
        key: start_command_factory(start_command, key, _context_settings=termui.CONTEXT_SETTINGS) for key in CONFIG_MAPPING
    },
    'start-grpc': {
        key: start_command_factory(start_grpc_command, key, _context_settings=termui.CONTEXT_SETTINGS, _serve_grpc=True) for key in CONFIG_MAPPING
    }
}
@cli.command(name='import', aliases=['download'])
@model_name_argument
@click.argument('model_id', type=click.STRING, default=None, metavar='Optional[REMOTE_REPO/MODEL_ID | /path/to/local/model]', required=False)
@click.argument('converter', envvar='CONVERTER', type=click.STRING, default=None, required=False, metavar=None)
@model_version_option
@output_option
@quantize_option
@machine_option
@backend_option
@serialisation_option
def import_command(
    model_name: str,
    model_id: str | None,
    converter: str | None,
    model_version: str | None,
    output: LiteralOutput,
    machine: bool,
    backend: LiteralBackend,
    quantize: LiteralQuantise | None,
    serialisation: t.Literal['safetensors', 'legacy'],
) -> bentoml.Model:
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
  $ openllm import opt facebook/opt-2.7b
  ```

  \b
  > If ``quantize`` is passed, the model weights will be saved as quantized weights. You should
  > only use this option if you want the weight to be quantized by default. Note that OpenLLM also
  > support on-demand quantisation during initial startup.

  \b
  ## Conversion strategies [EXPERIMENTAL]

  \b
  Some models will include built-in conversion strategies for specific weights format.
  It will be determined via the `CONVERTER` environment variable. Note that this envvar should only be use provisionally as it is not RECOMMENDED to export this
  and save to a ``.env`` file.

  The conversion strategies will have the following format and will be determined per architecture implementation:
  <base_format>-<target_format>

  \b
  For example: the below convert LlaMA-2 model format to hf:

  \b
  ```bash
  $ CONVERTER=llama2-hf openllm import llama /path/to/llama-2
  ```
  """
  llm_config = AutoConfig.for_model(model_name)
  env = EnvVarMixin(model_name, backend=llm_config.default_backend(), model_id=model_id, quantize=quantize)
  backend = first_not_none(backend, default=env['backend_value'])
  llm = infer_auto_class(backend).for_model(
      model_name, model_id=env['model_id_value'], llm_config=llm_config, model_version=model_version, ensure_available=False,
                                                                         quantize=env['quantize_value'],
serialisation=serialisation
  )
  _previously_saved = False
  try:
    _ref = openllm.serialisation.get(llm)
    _previously_saved = True
  except openllm.exceptions.OpenLLMException:
    if not machine and output == 'pretty':
      msg = f"'{model_name}' {'with model_id='+ model_id if model_id is not None else ''} does not exists in local store for backend {llm.__llm_backend__}. Saving to BENTOML_HOME{' (path=' + os.environ.get('BENTOML_HOME', BentoMLContainer.bentoml_home.get()) + ')' if get_debug_mode() else ''}..."
      termui.echo(msg, fg='yellow', nl=True)
    _ref = openllm.serialisation.get(llm, auto_import=True)
    if backend == 'pt' and is_torch_available() and torch.cuda.is_available(): torch.cuda.empty_cache()
  if machine: return _ref
  elif output == 'pretty':
    if _previously_saved: termui.echo(f"{model_name} with 'model_id={model_id}' is already setup for backend '{backend}': {_ref.tag!s}", nl=True, fg='yellow')
    else: termui.echo(f'Saved model: {_ref.tag}')
  elif output == 'json': termui.echo(orjson.dumps({'previously_setup': _previously_saved, 'backend': backend, 'tag': str(_ref.tag)}, option=orjson.OPT_INDENT_2).decode())
  else: termui.echo(_ref.tag)
  return _ref

@cli.command(context_settings={'token_normalize_func': inflection.underscore})
@model_name_argument
@model_id_option
@output_option
@machine_option
@backend_option
@click.option('--bento-version', type=str, default=None, help='Optional bento version for this BentoLLM. Default is the the model revision.')
@click.option('--overwrite', is_flag=True, help='Overwrite existing Bento for given LLM if it already exists.')
@workers_per_resource_option(factory=click, build=True)
@cog.optgroup.group(cls=cog.MutuallyExclusiveOptionGroup, name='Optimisation options')
@quantize_option(factory=cog.optgroup, build=True)
@click.option(
    '--enable-features',
    multiple=True,
    nargs=1,
    metavar='FEATURE[,FEATURE]',
    help='Enable additional features for building this LLM Bento. Available: {}'.format(', '.join(OPTIONAL_DEPENDENCIES))
)
@click.option(
    '--adapter-id',
    default=None,
    multiple=True,
    metavar='[PATH | [remote/][adapter_name:]adapter_id][, ...]',
    help="Optional adapters id to be included within the Bento. Note that if you are using relative path, '--build-ctx' must be passed."
)
@click.option('--build-ctx', help='Build context. This is required if --adapter-id uses relative path', default=None)
@model_version_option
@click.option('--dockerfile-template', default=None, type=click.File(), help='Optional custom dockerfile template to be used with this BentoLLM.')
@serialisation_option
@container_registry_option
@click.option(
    '--container-version-strategy', type=click.Choice(['release', 'latest', 'nightly']), default='release', help="Default container version strategy for the image from '--container-registry'"
)
@cog.optgroup.group(cls=cog.MutuallyExclusiveOptionGroup, name='Utilities options')
@cog.optgroup.option(
    '--containerize',
    default=False,
    is_flag=True,
    type=click.BOOL,
    help="Whether to containerize the Bento after building. '--containerize' is the shortcut of 'openllm build && bentoml containerize'."
)
@cog.optgroup.option('--push', default=False, is_flag=True, type=click.BOOL, help="Whether to push the result bento to BentoCloud. Make sure to login with 'bentoml cloud login' first.")
@click.option('--force-push', default=False, is_flag=True, type=click.BOOL, help='Whether to force push.')
@click.pass_context
def build_command(
    ctx: click.Context,
    /,
    model_name: str,
    model_id: str | None,
    bento_version: str | None,
    overwrite: bool,
    output: LiteralOutput,
    quantize: LiteralQuantise | None,
    enable_features: tuple[str, ...] | None,
    workers_per_resource: float | None,
    adapter_id: tuple[str, ...],
    build_ctx: str | None,
    backend: LiteralBackend,
    machine: bool,
    model_version: str | None,
    dockerfile_template: t.TextIO | None,
    containerize: bool,
    push: bool,
    serialisation: t.Literal['safetensors', 'legacy'],
    container_registry: LiteralContainerRegistry,
    container_version_strategy: LiteralContainerVersionStrategy,
    force_push: bool,
    **attrs: t.Any,
) -> bentoml.Bento:
  '''Package a given models into a Bento.

  \b
  ```bash
  $ openllm build flan-t5 --model-id google/flan-t5-large
  ```

  \b
  > [!NOTE]
  > To run a container built from this Bento with GPU support, make sure
  > to have https://github.com/NVIDIA/nvidia-container-toolkit install locally.

  \b
  > [!IMPORTANT]
  > To build the bento with compiled OpenLLM, make sure to prepend HATCH_BUILD_HOOKS_ENABLE=1. Make sure that the deployment
  > target also use the same Python version and architecture as build machine.
  '''
  if machine: output = 'porcelain'
  if enable_features: enable_features = tuple(itertools.chain.from_iterable((s.split(',') for s in enable_features)))

  _previously_built = False

  llm_config = AutoConfig.for_model(model_name)
  env = EnvVarMixin(model_name, backend=backend, model_id=model_id, quantize=quantize)

  # NOTE: We set this environment variable so that our service.py logic won't raise RuntimeError
  # during build. This is a current limitation of bentoml build where we actually import the service.py into sys.path
  try:
    os.environ.update({'OPENLLM_MODEL': inflection.underscore(model_name), 'OPENLLM_SERIALIZATION': serialisation, env.backend: env['backend_value']})
    if env['model_id_value']: os.environ[env.model_id] = str(env['model_id_value'])
    if env['quantize_value']: os.environ[env.quantize] = str(env['quantize_value'])

    llm = infer_auto_class(env['backend_value']).for_model(
        model_name, model_id=env['model_id_value'], llm_config=llm_config, ensure_available=True, model_version=model_version, quantize=env['quantize_value'], serialisation=serialisation, **attrs
    )

    labels = dict(llm.identifying_params)
    labels.update({'_type': llm.llm_type, '_framework': env['backend_value']})
    workers_per_resource = first_not_none(workers_per_resource, default=llm_config['workers_per_resource'])

    with fs.open_fs(f"temp://llm_{llm_config['model_name']}") as llm_fs:
      dockerfile_template_path = None
      if dockerfile_template:
        with dockerfile_template:
          llm_fs.writetext('Dockerfile.template', dockerfile_template.read())
        dockerfile_template_path = llm_fs.getsyspath('/Dockerfile.template')

      adapter_map: dict[str, str | None] | None = None
      if adapter_id:
        if not build_ctx: ctx.fail("'build_ctx' is required when '--adapter-id' is passsed.")
        adapter_map = {}
        for v in adapter_id:
          _adapter_id, *adapter_name = v.rsplit(':', maxsplit=1)
          name = adapter_name[0] if len(adapter_name) > 0 else None
          try:
            resolve_user_filepath(_adapter_id, build_ctx)
            src_folder_name = os.path.basename(_adapter_id)
            src_fs = fs.open_fs(build_ctx)
            llm_fs.makedir(src_folder_name, recreate=True)
            fs.copy.copy_dir(src_fs, _adapter_id, llm_fs, src_folder_name)
            adapter_map[src_folder_name] = name
          # this is the remote adapter, then just added back
          # note that there is a drawback here. If the path of the local adapter
          # path have the same name as the remote, then we currently don't support
          # that edge case.
          except FileNotFoundError:
            adapter_map[_adapter_id] = name
        os.environ['OPENLLM_ADAPTER_MAP'] = orjson.dumps(adapter_map).decode()

      _bento_version = first_not_none(bento_version, default=llm.tag.version)
      bento_tag = bentoml.Tag.from_taglike(f'{llm.llm_type}-service:{_bento_version}'.lower().strip())
      try:
        bento = bentoml.get(bento_tag)
        if overwrite:
          if output == 'pretty': termui.echo(f'Overwriting existing Bento {bento_tag}', fg='yellow')
          bentoml.delete(bento_tag)
          raise bentoml.exceptions.NotFound(f'Rebuilding existing Bento {bento_tag}') from None
        _previously_built = True
      except bentoml.exceptions.NotFound:
        bento = bundle.create_bento(
            bento_tag,
            llm_fs,
            llm,
            workers_per_resource=workers_per_resource,
            adapter_map=adapter_map,
            quantize=quantize,
            extra_dependencies=enable_features,
            dockerfile_template=dockerfile_template_path,
            container_registry=container_registry,
            container_version_strategy=container_version_strategy
        )
  except Exception as err:
    raise err from None

  if machine: termui.echo(f'__tag__:{bento.tag}', fg='white')
  elif output == 'pretty':
    if not get_quiet_mode() and (not push or not containerize):
      termui.echo('\n' + OPENLLM_FIGLET, fg='white')
      if not _previously_built: termui.echo(f'Successfully built {bento}.', fg='green')
      elif not overwrite: termui.echo(f"'{model_name}' already has a Bento built [{bento}]. To overwrite it pass '--overwrite'.", fg='yellow')
      termui.echo(
          '📖 Next steps:\n\n' + f"* Push to BentoCloud with 'bentoml push':\n\t$ bentoml push {bento.tag}\n\n" +
          f"* Containerize your Bento with 'bentoml containerize':\n\t$ bentoml containerize {bento.tag} --opt progress=plain\n\n" +
          "\tTip: To enable additional BentoML features for 'containerize', use '--enable-features=FEATURE[,FEATURE]' [see 'bentoml containerize -h' for more advanced usage]\n",
          fg='blue',
      )
  elif output == 'json':
    termui.echo(orjson.dumps(bento.info.to_dict(), option=orjson.OPT_INDENT_2).decode())
  else:
    termui.echo(bento.tag)

  if push: BentoMLContainer.bentocloud_client.get().push_bento(bento, context=t.cast(GlobalOptions, ctx.obj).cloud_context, force=force_push)
  elif containerize:
    container_backend = t.cast('DefaultBuilder', os.environ.get('BENTOML_CONTAINERIZE_BACKEND', 'docker'))
    try:
      bentoml.container.health(container_backend)
    except subprocess.CalledProcessError:
      raise OpenLLMException(f'Failed to use backend {backend}') from None
    try:
      bentoml.container.build(bento.tag, backend=container_backend, features=('grpc', 'io'))
    except Exception as err:
      raise OpenLLMException(f"Exception caught while containerizing '{bento.tag!s}':\n{err}") from err
  return bento

@cli.command()
@output_option
@click.option('--show-available', is_flag=True, default=False, help="Show available models in local store (mutually exclusive with '-o porcelain').")
@machine_option
@click.pass_context
def models_command(ctx: click.Context, output: LiteralOutput, show_available: bool, machine: bool) -> DictStrAny | None:
  """List all supported models.

  \b
  > NOTE: '--show-available' and '-o porcelain' are mutually exclusive.

  \b
  ```bash
  openllm models --show-available
  ```
  """
  from .._llm import normalise_model_name

  models = tuple(inflection.dasherize(key) for key in CONFIG_MAPPING.keys())
  if output == 'porcelain':
    if show_available: raise click.BadOptionUsage('--show-available', "Cannot use '--show-available' with '-o porcelain' (mutually exclusive).")
    termui.echo('\n'.join(models), fg='white')
  else:
    failed_initialized: list[tuple[str, Exception]] = []

    json_data: dict[str, dict[t.Literal['architecture', 'model_id', 'url', 'installation', 'cpu', 'gpu', 'backend'], t.Any] | t.Any] = {}
    converted: list[str] = []
    for m in models:
      config = AutoConfig.for_model(m)
      backend: tuple[str, ...] = ()
      if config['model_name'] in MODEL_MAPPING_NAMES: backend += ('pt',)
      if config['model_name'] in MODEL_FLAX_MAPPING_NAMES: backend += ('flax',)
      if config['model_name'] in MODEL_TF_MAPPING_NAMES: backend += ('tf',)
      if config['model_name'] in MODEL_VLLM_MAPPING_NAMES: backend += ('vllm',)
      json_data[m] = {
          'architecture': config['architecture'],
          'model_id': config['model_ids'],
          'cpu': not config['requires_gpu'],
          'gpu': True,
          'backend': backend,
          'installation': f'"openllm[{m}]"' if m in OPTIONAL_DEPENDENCIES or config['requirements'] else 'openllm',
      }
      converted.extend([normalise_model_name(i) for i in config['model_ids']])
      if DEBUG:
        try:
          AutoLLM.for_model(m, llm_config=config)
        except Exception as e:
          failed_initialized.append((m, e))

    ids_in_local_store = {
        k: [
            i for i in bentoml.models.list() if 'framework' in i.info.labels and i.info.labels['framework'] == 'openllm' and 'model_name' in i.info.labels and i.info.labels['model_name'] == k
        ] for k in json_data.keys()
    }
    ids_in_local_store = {k: v for k, v in ids_in_local_store.items() if v}
    local_models: DictStrAny | None = None
    if show_available:
      local_models = {k: [str(i.tag) for i in val] for k, val in ids_in_local_store.items()}

    if machine:
      if show_available: json_data['local'] = local_models
      return json_data
    elif output == 'pretty':
      import tabulate

      tabulate.PRESERVE_WHITESPACE = True
      # llm, architecture, url, model_id, installation, cpu, gpu, backend
      data: list[str | tuple[str, str, list[str], str, LiteralString, LiteralString, tuple[LiteralBackend, ...]]] = []
      for m, v in json_data.items():
        data.extend([(m, v['architecture'], v['model_id'], v['installation'], '❌' if not v['cpu'] else '✅', '✅', v['backend'],)])
      column_widths = [
          int(termui.COLUMNS / 12), int(termui.COLUMNS / 6), int(termui.COLUMNS / 4), int(termui.COLUMNS / 12), int(termui.COLUMNS / 12), int(termui.COLUMNS / 12), int(termui.COLUMNS / 4),
      ]

      if len(data) == 0 and len(failed_initialized) > 0:
        termui.echo('Exception found while parsing models:\n', fg='yellow')
        for m, err in failed_initialized:
          termui.echo(f'- {m}: ', fg='yellow', nl=False)
          termui.echo(traceback.print_exception(None, err, None, limit=5), fg='red')  # type: ignore[func-returns-value]
        sys.exit(1)

      table = tabulate.tabulate(data, tablefmt='fancy_grid', headers=['LLM', 'Architecture', 'Models Id', 'pip install', 'CPU', 'GPU', 'Runtime'], maxcolwidths=column_widths)
      termui.echo(table, fg='white')

      if DEBUG and len(failed_initialized) > 0:
        termui.echo('\nThe following models are supported but failed to initialize:\n')
        for m, err in failed_initialized:
          termui.echo(f'- {m}: ', fg='blue', nl=False)
          termui.echo(err, fg='red')

      if show_available:
        if len(ids_in_local_store) == 0:
          termui.echo('No models available locally.')
          ctx.exit(0)
        termui.echo('The following are available in local store:', fg='magenta')
        termui.echo(orjson.dumps(local_models, option=orjson.OPT_INDENT_2).decode(), fg='white')
    else:
      if show_available: json_data['local'] = local_models
      termui.echo(orjson.dumps(json_data, option=orjson.OPT_INDENT_2,).decode(), fg='white')
  ctx.exit(0)
@cli.command()
@model_name_argument(required=False)
@click.option('-y', '--yes', '--assume-yes', is_flag=True, help='Skip confirmation when deleting a specific model')
@click.option('--include-bentos/--no-include-bentos', is_flag=True, default=False, help='Whether to also include pruning bentos.')
@inject
def prune_command(
    model_name: str | None, yes: bool, include_bentos: bool, model_store: ModelStore = Provide[BentoMLContainer.model_store], bento_store: BentoStore = Provide[BentoMLContainer.bento_store]
) -> None:
  '''Remove all saved models, (and optionally bentos) built with OpenLLM locally.

  \b
  If a model type is passed, then only prune models for that given model type.
  '''
  available: list[tuple[bentoml.Model | bentoml.Bento,
                        ModelStore | BentoStore]] = [(m, model_store) for m in bentoml.models.list() if 'framework' in m.info.labels and m.info.labels['framework'] == 'openllm']
  if model_name is not None: available = [(m, store) for m, store in available if 'model_name' in m.info.labels and m.info.labels['model_name'] == inflection.underscore(model_name)]
  if include_bentos:
    if model_name is not None:
      available += [(b, bento_store) for b in bentoml.bentos.list() if 'start_name' in b.info.labels and b.info.labels['start_name'] == inflection.underscore(model_name)]
    else:
      available += [(b, bento_store) for b in bentoml.bentos.list() if '_type' in b.info.labels and '_framework' in b.info.labels]

  for store_item, store in available:
    if yes: delete_confirmed = True
    else: delete_confirmed = click.confirm(f"delete {'model' if isinstance(store, ModelStore) else 'bento'} {store_item.tag}?")
    if delete_confirmed:
      store.delete(store_item.tag)
      termui.echo(f"{store_item} deleted from {'model' if isinstance(store, ModelStore) else 'bento'} store.", fg='yellow')
def parsing_instruction_callback(ctx: click.Context, param: click.Parameter, value: list[str] | str | None) -> tuple[str, bool | str] | list[str] | str | None:
  if value is None:
    return value

  if isinstance(value, list):
    # we only parse --text foo bar -> --text foo and omit bar
    value = value[-1]

  key, *values = value.split('=')
  if not key.startswith('--'):
    raise click.BadParameter(f'Invalid option format: {value}')
  key = key[2:]
  if len(values) == 0:
    return key, True
  elif len(values) == 1:
    return key, values[0]
  else:
    raise click.BadParameter(f'Invalid option format: {value}')
def shared_client_options(f: _AnyCallable | None = None, output_value: t.Literal['json', 'porcelain', 'pretty'] = 'pretty') -> t.Callable[[FC], FC]:
  options = [
      click.option('--endpoint', type=click.STRING, help='OpenLLM Server endpoint, i.e: http://localhost:3000', envvar='OPENLLM_ENDPOINT', default='http://localhost:3000',
                   ),
      click.option('--timeout', type=click.INT, default=30, help='Default server timeout', show_default=True),
      output_option(default_value=output_value),
  ]
  return compose(*options)(f) if f is not None else compose(*options)
@cli.command()
@click.argument('task', type=click.STRING, metavar='TASK')
@shared_client_options
@click.option('--agent', type=click.Choice(['hf']), default='hf', help='Whether to interact with Agents from given Server endpoint.', show_default=True)
@click.option('--remote', is_flag=True, default=False, help='Whether or not to use remote tools (inference endpoints) instead of local ones.', show_default=True)
@click.option(
    '--opt',
    help="Define prompt options. "
    "(format: ``--opt text='I love this' --opt audio:./path/to/audio  --opt image:/path/to/file``)",
    required=False,
    multiple=True,
    callback=opt_callback,
    metavar='ARG=VALUE[,ARG=VALUE]'
)
def instruct_command(endpoint: str, timeout: int, agent: LiteralString, output: LiteralOutput, remote: bool, task: str, _memoized: DictStrAny, **attrs: t.Any) -> str:
  '''Instruct agents interactively for given tasks, from a terminal.

  \b
  ```bash
  $ openllm instruct --endpoint http://12.323.2.1:3000 \\
        "Is the following `text` (in Spanish) positive or negative?" \\
        --text "¡Este es un API muy agradable!"
  ```
  '''
  client = openllm.client.HTTPClient(endpoint, timeout=timeout)

  try:
    client.call('metadata')
  except http.client.BadStatusLine:
    raise click.ClickException(f'{endpoint} is neither a HTTP server nor reachable.') from None
  if agent == 'hf':
    _memoized = {k: v[0] for k, v in _memoized.items() if v}
    client._hf_agent.set_stream(logger.info)
    if output != 'porcelain': termui.echo(f"Sending the following prompt ('{task}') with the following vars: {_memoized}", fg='magenta')
    result = client.ask_agent(task, agent_type=agent, return_code=False, remote=remote, **_memoized)
    if output == 'json': termui.echo(orjson.dumps(result, option=orjson.OPT_INDENT_2).decode(), fg='white')
    else: termui.echo(result, fg='white')
    return result
  else:
    raise click.BadOptionUsage('agent', f'Unknown agent type {agent}')
@cli.command()
@shared_client_options(output_value='json')
@click.option('--server-type', type=click.Choice(['grpc', 'http']), help='Server type', default='http', show_default=True)
@click.argument('text', type=click.STRING, nargs=-1)
@machine_option
@click.pass_context
def embed_command(
    ctx: click.Context, text: tuple[str, ...], endpoint: str, timeout: int, server_type: t.Literal['http', 'grpc'], output: LiteralOutput, machine: bool
) -> EmbeddingsOutput | None:
  '''Get embeddings interactively, from a terminal.

  \b
  ```bash
  $ openllm embed --endpoint http://12.323.2.1:3000 "What is the meaning of life?" "How many stars are there in the sky?"
  ```
  '''
  client = openllm.client.HTTPClient(endpoint, timeout=timeout) if server_type == 'http' else openllm.client.GrpcClient(endpoint, timeout=timeout)
  try:
    gen_embed = client.embed(text)
  except ValueError:
    raise click.ClickException(f'Endpoint {endpoint} does not support embeddings.') from None
  if machine: return gen_embed
  elif output == 'pretty':
    termui.echo('Generated embeddings: ', fg='magenta', nl=False)
    termui.echo(gen_embed.embeddings, fg='white')
    termui.echo('\nNumber of tokens: ', fg='magenta', nl=False)
    termui.echo(gen_embed.num_tokens, fg='white')
  elif output == 'json':
    termui.echo(orjson.dumps(bentoml_cattr.unstructure(gen_embed), option=orjson.OPT_INDENT_2).decode(), fg='white')
  else:
    termui.echo(gen_embed.embeddings, fg='white')
  ctx.exit(0)
@cli.command()
@shared_client_options
@click.option('--server-type', type=click.Choice(['grpc', 'http']), help='Server type', default='http', show_default=True)
@click.argument('prompt', type=click.STRING)
@click.option(
    '--sampling-params', help='Define query options. (format: ``--opt temperature=0.8 --opt=top_k:12)', required=False, multiple=True, callback=opt_callback, metavar='ARG=VALUE[,ARG=VALUE]'
)
@click.pass_context
def query_command(
    ctx: click.Context, /, prompt: str, endpoint: str, timeout: int, server_type: t.Literal['http', 'grpc'], output: LiteralOutput, _memoized: DictStrAny, **attrs: t.Any
) -> None:
  '''Ask a LLM interactively, from a terminal.

  \b
  ```bash
  $ openllm query --endpoint http://12.323.2.1:3000 "What is the meaning of life?"
  ```
  '''
  _memoized = {k: orjson.loads(v[0]) for k, v in _memoized.items() if v}
  if server_type == 'grpc': endpoint = re.sub(r'http://', '', endpoint)
  client = openllm.client.HTTPClient(endpoint, timeout=timeout) if server_type == 'http' else openllm.client.GrpcClient(endpoint, timeout=timeout)
  input_fg, generated_fg = 'magenta', 'cyan'
  if output != 'porcelain':
    termui.echo('==Input==\n', fg='white')
    termui.echo(f'{prompt}', fg=input_fg)
  res = client.query(prompt, return_response='raw', **{**client.configuration, **_memoized})
  if output == 'pretty':
    response = client.config.postprocess_generate(prompt, res['responses'])
    termui.echo('\n\n==Responses==\n', fg='white')
    termui.echo(response, fg=generated_fg)
  elif output == 'json':
    termui.echo(orjson.dumps(res, option=orjson.OPT_INDENT_2).decode(), fg='white')
  else:
    termui.echo(res['responses'], fg='white')
  ctx.exit(0)
@cli.group(cls=Extensions, hidden=True, name='extension')
def extension_command() -> None:
  '''Extension for OpenLLM CLI.'''
if __name__ == '__main__': cli()
