from __future__ import annotations

import importlib.metadata, os, platform, random, sys, typing
import questionary, typer

from collections import defaultdict
from openllm.accelerator_spec import can_run, get_local_machine_spec
from openllm.analytic import DO_NOT_TRACK, OpenLLMTyper
from openllm.clean import app as clean_app
from openllm.cloud import deploy as cloud_deploy, ensure_cloud_context, get_cloud_machine_spec
from openllm.common import CHECKED, INTERACTIVE, VERBOSE_LEVEL, BentoInfo, output
from openllm.local import run as local_run, serve as local_serve
from openllm.model import app as model_app, ensure_bento, list_bento
from openllm.repo import app as repo_app, cmd_update

if typing.TYPE_CHECKING:
  from openllm.common import DeploymentTarget

app = OpenLLMTyper(
  help='`openllm hello` to get started. '
  'OpenLLM is a CLI tool to manage and deploy open source LLMs and'
  ' get an OpenAI API compatible chat server in seconds.'
)

app.add_typer(repo_app, name='repo')
app.add_typer(model_app, name='model')
app.add_typer(clean_app, name='clean')


def _select_bento_name(models: list[BentoInfo], target: DeploymentTarget) -> tuple[str, str]:
  from tabulate import tabulate

  model_infos = [(model.repo.name, model.name, can_run(model, target)) for model in models]
  model_name_groups: defaultdict[tuple[str, str], float] = defaultdict(lambda: 0.0)
  for repo, name, score in model_infos:
    model_name_groups[repo, name] += score
  table_data = [
    (name, repo, CHECKED if score > 0 else '') for (repo, name), score in model_name_groups.items()
  ]
  if not table_data:
    output('No model found', style='red')
    raise typer.Exit(1)
  table: list[str] = tabulate(table_data, headers=['model', 'repo', 'locally runnable']).split('\n')

  selected: tuple[str, str] | None = questionary.select(
    'Select a model',
    [
      questionary.Separator(f'{table[0]}\n   {table[1]}'),
      *[questionary.Choice(line, value=value[:2]) for value, line in zip(table_data, table[2:])],
    ],
  ).ask()
  if selected is None:
    raise typer.Exit(1)
  return selected


def _select_bento_version(
  models: list[BentoInfo], target: DeploymentTarget | None, bento_name: str, repo: str
) -> tuple[BentoInfo, float]:
  from tabulate import tabulate

  model_infos: list[tuple[BentoInfo, float]] = [
    (model, can_run(model, target))
    for model in models
    if model.name == bento_name and model.repo.name == repo
  ]

  table_data = [
    [model.tag, CHECKED if score > 0 else '']
    for model, score in model_infos
    if model.name == bento_name and model.repo.name == repo
  ]
  if not table_data:
    output(f'No model found for {bento_name} in {repo}', style='red')
    raise typer.Exit(1)
  table: list[str] = tabulate(table_data, headers=['version', 'locally runnable']).split('\n')

  selected: tuple[BentoInfo, float] | None = questionary.select(
    'Select a version',
    [
      questionary.Separator(f'{table[0]}\n   {table[1]}'),
      *[questionary.Choice(line, value=value[:2]) for value, line in zip(model_infos, table[2:])],
    ],
  ).ask()
  if selected is None:
    raise typer.Exit(1)
  return selected


def _select_target(bento: BentoInfo, targets: list[DeploymentTarget]) -> DeploymentTarget:
  from tabulate import tabulate

  targets.sort(key=lambda x: can_run(bento, x), reverse=True)
  if not targets:
    output('No available instance type, check your bentocloud account', style='red')
    raise typer.Exit(1)

  table = tabulate(
    [
      [
        target.name,
        target.accelerators_repr,
        f'${target.price}',
        CHECKED if can_run(bento, target) else 'insufficient res.',
      ]
      for target in targets
    ],
    headers=['instance type', 'accelerator', 'price/hr', 'deployable'],
  ).split('\n')

  selected: DeploymentTarget | None = questionary.select(
    'Select an instance type',
    [
      questionary.Separator(f'{table[0]}\n   {table[1]}'),
      *[questionary.Choice(f'{line}', value=target) for target, line in zip(targets, table[2:])],
    ],
  ).ask()
  if selected is None:
    raise typer.Exit(1)
  return selected


def _select_action(bento: BentoInfo, score: float, context: typing.Optional[str] = None) -> None:
  if score > 0:
    options: list[typing.Any] = [
      questionary.Separator('Available actions'),
      questionary.Choice('0. Run the model in terminal', value='run', shortcut_key='0'),
      questionary.Separator(f'  $ openllm run {bento}'),
      questionary.Separator(' '),
      questionary.Choice(
        '1. Serve the model locally and get a chat server', value='serve', shortcut_key='1'
      ),
      questionary.Separator(f'  $ openllm serve {bento}'),
      questionary.Separator(' '),
      questionary.Choice(
        '2. Deploy the model to bentocloud and get a scalable chat server',
        value='deploy',
        shortcut_key='2',
      ),
      questionary.Separator(f'  $ openllm deploy {bento}'),
    ]
  else:
    options = [
      questionary.Separator('Available actions'),
      questionary.Choice(
        '0. Run the model in terminal', value='run', disabled='insufficient res.', shortcut_key='0'
      ),
      questionary.Separator(f'  $ openllm run {bento}'),
      questionary.Separator(' '),
      questionary.Choice(
        '1. Serve the model locally and get a chat server',
        value='serve',
        disabled='insufficient res.',
        shortcut_key='1',
      ),
      questionary.Separator(f'  $ openllm serve {bento}'),
      questionary.Separator(' '),
      questionary.Choice(
        '2. Deploy the model to bentocloud and get a scalable chat server',
        value='deploy',
        shortcut_key='2',
      ),
      questionary.Separator(f'  $ openllm deploy {bento}'),
    ]
  action: str | None = questionary.select('Select an action', options).ask()
  if action is None:
    raise typer.Exit(1)
  if action == 'run':
    try:
      port = random.randint(30000, 40000)
      local_run(bento, port=port)
    finally:
      output('\nUse this command to run the action again:', style='green')
      output(f'  $ openllm run {bento}', style='orange')
  elif action == 'serve':
    try:
      local_serve(bento)
    finally:
      output('\nUse this command to run the action again:', style='green')
      output(f'  $ openllm serve {bento}', style='orange')
  elif action == 'deploy':
    ensure_cloud_context()
    targets = get_cloud_machine_spec(context=context)
    target = _select_target(bento, targets)
    try:
      cloud_deploy(bento, target, context=context)
    finally:
      output('\nUse this command to run the action again:', style='green')
      output(f'  $ openllm deploy {bento} --instance-type {target.name}', style='orange')


@app.command(help='get started interactively')
def hello(
  repo: typing.Optional[str] = None,
  env: typing.Optional[list[str]] = typer.Option(
    None,
    '--env',
    help='Environment variables to pass to the deployment command. Format: NAME or NAME=value. Can be specified multiple times.',
  ),
  context: typing.Optional[str] = typer.Option(
    None, '--context', help='BentoCloud context name to pass to the deployment command.'
  ),
) -> None:
  cmd_update()
  INTERACTIVE.set(True)

  target = get_local_machine_spec()
  output(f'  Detected Platform: {target.platform}', style='green')
  if target.accelerators:
    output('  Detected Accelerators: ', style='green')
    for a in target.accelerators:
      output(f'   - {a.model} {a.memory_size}GB', style='green')
  else:
    output('  Detected Accelerators: None', style='green')

  models = list_bento(repo_name=repo)
  if not models:
    output('No model found, you probably need to update the model repo:', style='red')
    output('  $ openllm repo update', style='orange')
    raise typer.Exit(1)

  bento_name, repo = _select_bento_name(models, target)
  bento, score = _select_bento_version(models, target, bento_name, repo)
  _select_action(bento, score, context=context)


@app.command(help='start an OpenAI API compatible chat server and chat in browser')
def serve(
  model: typing.Annotated[str, typer.Argument()],
  repo: typing.Optional[str] = None,
  port: int = 3000,
  verbose: bool = False,
  env: typing.Optional[list[str]] = typer.Option(
    None,
    '--env',
    help='Environment variables to pass to the deployment command. Format: NAME or NAME=value. Can be specified multiple times.',
  ),
  arg: typing.Optional[list[str]] = typer.Option(
    None,
    '--arg',
    help='Bento arguments in the form of key=value pairs. Can be specified multiple times.',
  ),
) -> None:
  cmd_update()
  if verbose:
    VERBOSE_LEVEL.set(20)
  target = get_local_machine_spec()
  bento = ensure_bento(model, target=target, repo_name=repo)
  local_serve(bento, port=port, cli_envs=env, cli_args=arg)


@app.command(help='run the model and chat in terminal')
def run(
  model: typing.Annotated[str, typer.Argument()] = '',
  repo: typing.Optional[str] = None,
  port: typing.Optional[int] = None,
  timeout: int = 600,
  verbose: bool = False,
  env: typing.Optional[list[str]] = typer.Option(
    None,
    '--env',
    help='Environment variables to pass to the deployment command. Format: NAME or NAME=value. Can be specified multiple times.',
  ),
  arg: typing.Optional[list[str]] = typer.Option(
    None,
    '--arg',
    help='Bento arguments in the form of key=value pairs. Can be specified multiple times.',
  ),
) -> None:
  cmd_update()
  if verbose:
    VERBOSE_LEVEL.set(20)
  target = get_local_machine_spec()
  bento = ensure_bento(model, target=target, repo_name=repo)
  if port is None:
    port = random.randint(30000, 40000)
  local_run(bento, port=port, timeout=timeout, cli_envs=env, cli_args=arg)


@app.command(help='deploy production-ready OpenAI API-compatible server to BentoCloud')
def deploy(
  model: typing.Annotated[str, typer.Argument()] = '',
  instance_type: typing.Optional[str] = None,
  repo: typing.Optional[str] = None,
  verbose: bool = False,
  env: typing.Optional[list[str]] = typer.Option(
    None,
    '--env',
    help='Environment variables to pass to the deployment command. Format: NAME or NAME=value. Can be specified multiple times.',
  ),
  context: typing.Optional[str] = typer.Option(
    None, '--context', help='BentoCloud context name to pass to the deployment command.'
  ),
) -> None:
  cmd_update()
  if verbose:
    VERBOSE_LEVEL.set(20)
  bento = ensure_bento(model, repo_name=repo)
  if instance_type is not None:
    return cloud_deploy(
      bento, DeploymentTarget(accelerators=[], name=instance_type), cli_envs=env, context=context
    )
  targets = sorted(
    filter(lambda x: can_run(bento, x) > 0, get_cloud_machine_spec(context=context)),
    key=lambda x: can_run(bento, x),
    reverse=True,
  )
  if not targets:
    output('No available instance type, check your bentocloud account', style='red')
    raise typer.Exit(1)
  target = targets[0]
  output(f'Recommended instance type: {target.name}', style='green')
  cloud_deploy(bento, target, cli_envs=env, context=context)


@app.callback(invoke_without_command=True)
def typer_callback(
  verbose: int = 0,
  do_not_track: bool = typer.Option(
    False, '--do-not-track', help='Whether to disable usage tracking', envvar=DO_NOT_TRACK
  ),
  version: bool = typer.Option(False, '--version', '-v', help='Show version'),
) -> None:
  if verbose:
    VERBOSE_LEVEL.set(verbose)
  if version:
    output(
      f'openllm, {importlib.metadata.version("openllm")}\nPython ({platform.python_implementation()}) {platform.python_version()}'
    )
    sys.exit(0)
  if do_not_track:
    os.environ[DO_NOT_TRACK] = str(True)


if __name__ == '__main__':
  app()
