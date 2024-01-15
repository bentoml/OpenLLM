from __future__ import annotations
import typing as t

import click
import inflection
import orjson

import bentoml
import openllm
from bentoml._internal.utils import human_readable_size
from openllm_cli import termui
from openllm_cli._factory import model_complete_envvar, model_name_argument

if t.TYPE_CHECKING:
  from openllm_core._typing_compat import DictStrAny


@click.command('list_models', context_settings=termui.CONTEXT_SETTINGS)
@model_name_argument(required=False, shell_complete=model_complete_envvar)
def cli(model_name: str | None) -> DictStrAny:
  """List available models in local store to be used with OpenLLM."""
  models = tuple(inflection.dasherize(key) for key in openllm.CONFIG_MAPPING.keys())
  ids_in_local_store = {
    k: [
      i
      for i in bentoml.models.list()
      if 'framework' in i.info.labels
      and i.info.labels['framework'] == 'openllm'
      and 'model_name' in i.info.labels
      and i.info.labels['model_name'] == k
    ]
    for k in models
  }
  if model_name is not None:
    ids_in_local_store = {
      k: [i for i in v if 'model_name' in i.info.labels and i.info.labels['model_name'] == inflection.dasherize(model_name)]
      for k, v in ids_in_local_store.items()
    }
  ids_in_local_store = {k: v for k, v in ids_in_local_store.items() if v}
  local_models = {
    k: [{'tag': str(i.tag), 'size': human_readable_size(openllm.utils.calc_dir_size(i.path))} for i in val] for k, val in ids_in_local_store.items()
  }
  termui.echo(orjson.dumps(local_models, option=orjson.OPT_INDENT_2).decode(), fg='white')
  return local_models
