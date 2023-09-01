from __future__ import annotations
import importlib.machinery
import logging
import os
import pkgutil
import subprocess
import sys
import tempfile
import typing as t

import click
import yaml

from openllm import playground
from openllm.cli import termui
from openllm_core.utils import is_jupyter_available
from openllm_core.utils import is_jupytext_available
from openllm_core.utils import is_notebook_available

if t.TYPE_CHECKING:
  import jupytext
  import nbformat

  from openllm_core._typing_compat import DictStrAny

logger = logging.getLogger(__name__)

def load_notebook_metadata() -> DictStrAny:
  with open(os.path.join(os.path.dirname(playground.__file__), '_meta.yml'), 'r') as f:
    content = yaml.safe_load(f)
  if not all('description' in k for k in content.values()):
    raise ValueError("Invalid metadata file. All entries must have a 'description' key.")
  return content

@click.command('playground', context_settings=termui.CONTEXT_SETTINGS)
@click.argument('output-dir', default=None, required=False)
@click.option('--port', envvar='JUPYTER_PORT', show_envvar=True, show_default=True, default=8888, help='Default port for Jupyter server')
@click.pass_context
def cli(ctx: click.Context, output_dir: str | None, port: int) -> None:
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
  > [!NOTE]
  > This command requires Jupyter to be installed. Install it with 'pip install "openllm[playground]"'
  """
  if not is_jupyter_available() or not is_jupytext_available() or not is_notebook_available():
    raise RuntimeError("Playground requires 'jupyter', 'jupytext', and 'notebook'. Install it with 'pip install \"openllm[playground]\"'")
  metadata = load_notebook_metadata()
  _temp_dir = False
  if output_dir is None:
    _temp_dir = True
    output_dir = tempfile.mkdtemp(prefix='openllm-playground-')
  else:
    os.makedirs(os.path.abspath(os.path.expandvars(os.path.expanduser(output_dir))), exist_ok=True)

  termui.echo('The playground notebooks will be saved to: ' + os.path.abspath(output_dir), fg='blue')
  for module in pkgutil.iter_modules(playground.__path__):
    if module.ispkg or os.path.exists(os.path.join(output_dir, module.name + '.ipynb')):
      logger.debug('Skipping: %s (%s)', module.name, 'File already exists' if not module.ispkg else f'{module.name} is a module')
      continue
    if not isinstance(module.module_finder, importlib.machinery.FileFinder): continue
    termui.echo('Generating notebook for: ' + module.name, fg='magenta')
    markdown_cell = nbformat.v4.new_markdown_cell(metadata[module.name]['description'])
    f = jupytext.read(os.path.join(module.module_finder.path, module.name + '.py'))
    f.cells.insert(0, markdown_cell)
    jupytext.write(f, os.path.join(output_dir, module.name + '.ipynb'), fmt='notebook')
  try:
    subprocess.check_output([sys.executable, '-m', 'jupyter', 'notebook', '--notebook-dir', output_dir, '--port', str(port), '--no-browser', '--debug'])
  except subprocess.CalledProcessError as e:
    termui.echo(e.output, fg='red')
    raise click.ClickException(f'Failed to start a jupyter server:\n{e}') from None
  except KeyboardInterrupt:
    termui.echo('\nShutting down Jupyter server...', fg='yellow')
    if _temp_dir: termui.echo('Note: You can access the generated notebooks in: ' + output_dir, fg='blue')
  ctx.exit(0)
