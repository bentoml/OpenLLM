#!/usr/bin/env python3
from __future__ import annotations
import typing as t
from ghapi.all import GhApi
from pathlib import Path
from plumbum.cmd import curl, shasum, cut

if t.TYPE_CHECKING:
  from openllm._types import DictStrAny
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
  return 0

if __name__ == "__main__": raise SystemExit(main())
