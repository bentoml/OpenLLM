#!/usr/bin/env python3
from __future__ import annotations
import typing as t

from jinja2 import Environment
from jinja2.loaders import FileSystemLoader
from ghapi.all import GhApi
from pathlib import Path
from plumbum.cmd import curl, shasum, cut

if t.TYPE_CHECKING:
  from plumbum.commands.base import Pipeline

# get git root from this file
ROOT = Path(__file__).parent

_OWNER = "bentoml"
_REPO = "openllm"

_gz_strategies = {"macos_arm": "aarch64-apple-darwin", "macos_intel": "x86_64-apple-darwin", "linux_intel": "x86_64-unknown-linux-musl"}

def determine_release_url(svn_url: str, tag: str, target: t.Literal["macos_arm", "macos_intel", "linux_intel", "archive"]) -> str:
  if target == 'archive': return f"{svn_url}/archive/{tag}.tar.gz"
  return f"{svn_url}/releases/download/openllm-{tag.replace('v', '')}-{_gz_strategies[target]}.tar.gz"

# curl -sSL <svn_url>/archive/refs/tags/<tag>.tar.gz | shasum -a256 | cut -d'' -f1
def get_release_hash_command(svn_url: str, tag: str) -> Pipeline:
  return curl["-sSL", svn_url] | shasum["-a256"] | cut["-d' '", "-f1"]

def main() -> int:
  api = GhApi(owner=_OWNER, repo=_REPO)
  _info = api.repos.get()
  tags = api.repos.get_latest_release().name

  breakpoint()
  shadict = {k: get_release_hash_command(determine_release_url(_info.svn_url, "v0.2.12", k), "v0.2.12")() for k in _gz_strategies}

  ENVIRONMENT = Environment(extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols", "jinja2.ext.debug"], trim_blocks=True, lstrip_blocks=True, loader=FileSystemLoader((ROOT / "Formula").__fspath__(), followlinks=True),)
  with (ROOT / "Formula" / "openllm.rb").open("w") as f:
    f.write(ENVIRONMENT.get_template("openllm.rb.j2", globals={"determine_release_url": determine_rgelease_url}).render(**_info, release_url=determine_release_url(_info.svn_url, _info.tag_name, "archive"), release_hash=get_release_hash_command(_info.svn_url, _info.tag_name),))
  return 0

if __name__ == "__main__": raise SystemExit(main())
