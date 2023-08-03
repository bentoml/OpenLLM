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
import os
import typing as t

import click
import inflection

from ..utils import get_debug_mode
from ..utils import get_quiet_mode

if t.TYPE_CHECKING:
  from .._types import DictStrAny

def echo(text: t.Any, fg: str = "green", _with_style: bool = True, **attrs: t.Any) -> None:
  attrs["fg"] = fg if not get_debug_mode() else None
  if not get_quiet_mode(): t.cast(t.Callable[..., None], click.echo if not _with_style else click.secho)(text, **attrs)

COLUMNS: int = int(os.environ.get("COLUMNS", str(120)))

CONTEXT_SETTINGS: DictStrAny = {"help_option_names": ["-h", "--help"], "max_content_width": COLUMNS, "token_normalize_func": inflection.underscore}

__all__ = ["echo", "COLUMNS", "CONTEXT_SETTINGS"]
