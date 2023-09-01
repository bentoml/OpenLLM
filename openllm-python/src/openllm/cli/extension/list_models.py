from __future__ import annotations
import typing as t

import click
import inflection
import orjson

import bentoml
import openllm

from bentoml._internal.utils import human_readable_size
from openllm.cli import termui
from openllm.cli._factory import LiteralOutput
from openllm.cli._factory import model_complete_envvar
from openllm.cli._factory import model_name_argument
from openllm.cli._factory import output_option

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import DictStrAny

@click.command('list_models', context_settings=termui.CONTEXT_SETTINGS)
@model_name_argument(required=False, shell_complete=model_complete_envvar)
@output_option(default_value='json')
def cli(model_name: str | None, output: LiteralOutput) -> DictStrAny:
  '''This is equivalent to openllm models --show-available less the nice table.'''
  models = tuple(inflection.dasherize(key) for key in openllm.CONFIG_MAPPING.keys())
  ids_in_local_store = {
      k: [i for i in bentoml.models.list() if 'framework' in i.info.labels and i.info.labels['framework'] == 'openllm' and 'model_name' in i.info.labels and i.info.labels['model_name'] == k]
      for k in models
  }
  if model_name is not None:
    ids_in_local_store = {k: [i for i in v if 'model_name' in i.info.labels and i.info.labels['model_name'] == inflection.dasherize(model_name)] for k, v in ids_in_local_store.items()}
  ids_in_local_store = {k: v for k, v in ids_in_local_store.items() if v}
  local_models = {k: [{'tag': str(i.tag), 'size': human_readable_size(openllm.utils.calc_dir_size(i.path))} for i in val] for k, val in ids_in_local_store.items()}
  if output == 'pretty':
    import tabulate
    tabulate.PRESERVE_WHITESPACE = True
    termui.echo(tabulate.tabulate([(k, i['tag'], i['size']) for k, v in local_models.items() for i in v], tablefmt='fancy_grid', headers=['LLM', 'Tag', 'Size']), fg='white')
  else:
    termui.echo(orjson.dumps(local_models, option=orjson.OPT_INDENT_2).decode(), fg='white')
  return local_models
