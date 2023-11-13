from __future__ import annotations

import click
import inflection
import orjson

import bentoml
import openllm
from bentoml._internal.utils import human_readable_size
from openllm_cli import termui


@click.command('list_bentos', context_settings=termui.CONTEXT_SETTINGS)
@click.pass_context
def cli(ctx: click.Context) -> None:
  """List available bentos built by OpenLLM."""
  mapping = {
    k: [
      {
        'tag': str(b.tag),
        'size': human_readable_size(openllm.utils.calc_dir_size(b.path)),
        'models': [
          {'tag': str(m.tag), 'size': human_readable_size(openllm.utils.calc_dir_size(m.path))}
          for m in (bentoml.models.get(_.tag) for _ in b.info.models)
        ],
      }
      for b in tuple(i for i in bentoml.list() if all(k in i.info.labels for k in {'start_name', 'bundler'}))
      if b.info.labels['start_name'] == k
    ]
    for k in tuple(inflection.dasherize(key) for key in openllm.CONFIG_MAPPING.keys())
  }
  mapping = {k: v for k, v in mapping.items() if v}
  termui.echo(orjson.dumps(mapping, option=orjson.OPT_INDENT_2).decode(), fg='white')
  ctx.exit(0)
