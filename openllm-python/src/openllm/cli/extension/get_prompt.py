from __future__ import annotations
import logging
import traceback
import typing as t

import click
import inflection
import orjson

from bentoml_cli.utils import opt_callback

import openllm
import openllm_core

from openllm.cli import termui
from openllm.cli._factory import model_complete_envvar
from openllm_core.prompts import process_prompt

logger = logging.getLogger(__name__)

@click.command('get_prompt', context_settings=termui.CONTEXT_SETTINGS)
@click.argument('model_name', type=click.Choice([inflection.dasherize(name) for name in openllm.CONFIG_MAPPING.keys()]), shell_complete=model_complete_envvar)
@click.argument('prompt', type=click.STRING)
@click.option('--format', type=click.STRING, default=None)
@click.option('--opt',
              help="Define additional prompt variables. (format: ``--opt system_prompt='You are a useful assistant'``)",
              required=False,
              multiple=True,
              callback=opt_callback,
              metavar='ARG=VALUE[,ARG=VALUE]')
@click.pass_context
def cli(ctx: click.Context, /, model_name: str, prompt: str, format: str | None, _memoized: dict[str, t.Any], **_: t.Any) -> str | None:
  """Get the default prompt used by OpenLLM."""
  module = getattr(openllm_core.config, f'configuration_{model_name}')
  _memoized = {k: v[0] for k, v in _memoized.items() if v}
  try:
    template = getattr(module, 'DEFAULT_PROMPT_TEMPLATE', None)
    prompt_mapping = getattr(module, 'PROMPT_MAPPING', None)
    if template is None:
      raise click.BadArgumentUsage(f'model {model_name} does not have a default prompt template') from None
    if callable(template):
      if format is None:
        if not hasattr(module, 'PROMPT_MAPPING') or module.PROMPT_MAPPING is None:
          raise RuntimeError('Failed to find prompt mapping while DEFAULT_PROMPT_TEMPLATE is a function.')
        raise click.BadOptionUsage('format', f"{model_name} prompt requires passing '--format' (available format: {list(module.PROMPT_MAPPING)})")
      if prompt_mapping is None:
        raise click.BadArgumentUsage(f'Failed to fine prompt mapping while the default prompt for {model_name} is a callable.') from None
      if format not in prompt_mapping:
        raise click.BadOptionUsage('format', f'Given format {format} is not valid for {model_name} (available format: {list(prompt_mapping)})')
      _prompt_template = template(format)
    else:
      _prompt_template = template
    try:
      # backward-compatible. TO BE REMOVED once every model has default system message and prompt template.
      fully_formatted = process_prompt(prompt, _prompt_template, True, **_memoized)
    except RuntimeError as err:
      logger.debug('Exception caught while formatting prompt: %s', err)
      fully_formatted = openllm.AutoConfig.for_model(model_name).sanitize_parameters(prompt, prompt_template=_prompt_template)[0]
    termui.echo(orjson.dumps({'prompt': fully_formatted}, option=orjson.OPT_INDENT_2).decode(), fg='white')
  except Exception as err:
    traceback.print_exc()
    raise click.ClickException(f'Failed to determine a default prompt template for {model_name}.') from err
  ctx.exit(0)
