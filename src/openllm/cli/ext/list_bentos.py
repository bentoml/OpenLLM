from __future__ import annotations

import click
import inflection
import orjson

import bentoml
import openllm

from .. import termui

@click.command("list_bentos", context_settings=termui.CONTEXT_SETTINGS)
@click.pass_context
def cli(ctx: click.Context) -> None:
    """List available bentos built by OpenLLM."""
    _local_bentos = {str(i.tag): i.info.labels["start_name"] for i in bentoml.list() if "start_name" in i.info.labels}
    mapping = {k: [tag for tag, name in _local_bentos.items() if name == k] for k in tuple(inflection.dasherize(key) for key in openllm.CONFIG_MAPPING.keys())}
    mapping = {k: v for k, v in mapping.items() if v}
    termui.echo(orjson.dumps(mapping, option=orjson.OPT_INDENT_2).decode(), fg="white")
    ctx.exit(0)
