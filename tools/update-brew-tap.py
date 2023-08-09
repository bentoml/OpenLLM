#!/usr/bin/env python3
# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import typing as t

import os
import fs
from jinja2 import Environment
from jinja2.loaders import FileSystemLoader
from ghapi.all import GhApi
from pathlib import Path
from plumbum.cmd import curl, shasum, cut

if t.TYPE_CHECKING:
  from openllm._types import DictStrAny
  from plumbum.commands.base import Pipeline

# get git root from this file
ROOT = Path(__file__).parent.parent

_OWNER = "bentoml"
_REPO = "openllm"

_gz_strategies: dict[t.Literal["macos_arm", "macos_intel", "linux_intel"], str] = {"macos_arm": "aarch64-apple-darwin", "macos_intel": "x86_64-apple-darwin", "linux_intel": "x86_64-unknown-linux-musl"}

def determine_release_url(svn_url: str, tag: str, target: t.Literal["macos_arm", "macos_intel", "linux_intel", "archive"]) -> str:
  if target == 'archive': return f"{svn_url}/archive/{tag}.tar.gz"
  return f"{svn_url}/releases/download/{tag}/openllm-{tag.replace('v', '')}-{_gz_strategies[target]}.tar.gz"

# curl -sSL <svn_url>/archive/refs/tags/<tag>.tar.gz | shasum -a256 | cut -d'' -f1
def get_release_hash_command(svn_url: str, tag: str) -> Pipeline:
  return curl["-sSL", svn_url] | shasum["-a256"] | cut["-d", " ", "-f1"]

def main() -> int:
  api = GhApi(owner=_OWNER, repo=_REPO, authenticate=False)
  _info = api.repos.get()
  release_tag = api.repos.get_latest_release().name

  shadict: DictStrAny = {k: get_release_hash_command(determine_release_url(_info.svn_url, release_tag, k), release_tag)().strip() for k in _gz_strategies}
  shadict['archive'] = get_release_hash_command(determine_release_url(_info.svn_url, release_tag, "archive"), release_tag)().strip()

  ENVIRONMENT = Environment(extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols", "jinja2.ext.debug"], trim_blocks=True, lstrip_blocks=True, loader=FileSystemLoader((ROOT / "Formula").__fspath__(), followlinks=True))
  template_file = "openllm.rb.j2"
  with (ROOT / "Formula" / "openllm.rb").open("w") as f:
    f.write(ENVIRONMENT.get_template(template_file, globals={"determine_release_url": determine_release_url}).render(shadict=shadict, __tag__=release_tag, __cmd__=fs.path.join(os.path.basename(os.path.dirname(__file__)), os.path.basename(__file__)), __template_file__=fs.path.join("Formula", template_file), __gz_extension__=_gz_strategies, **_info))
    f.write("\n")
  return 0

if __name__ == "__main__": raise SystemExit(main())
