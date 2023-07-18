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
import string
import typing as t


class PromptFormatter(string.Formatter):
    """This PromptFormatter is largely based on langchain's implementation."""

    def vformat(self, format_string: str, args: t.Sequence[t.Any], kwargs: t.Mapping[str, t.Any]) -> t.Any:
        if len(args) > 0:
            raise ValueError("Positional arguments are not supported")
        return super().vformat(format_string, args, kwargs)

    def check_unused_args(
        self, used_args: set[int | str], args: t.Sequence[t.Any], kwargs: t.Mapping[str, t.Any]
    ) -> None:
        """Check if extra params is passed."""
        extras = set(kwargs).difference(used_args)
        if extras:
            raise KeyError(f"Extra params passed: {extras}")

    def extract_template_variables(self, template: str) -> t.Sequence[str]:
        """Extract template variables from a template string."""
        return [field[1] for field in self.parse(template) if field[1] is not None]


default_formatter = PromptFormatter()
