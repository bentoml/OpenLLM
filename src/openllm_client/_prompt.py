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

import attr

import openllm
from openllm._prompt import PromptFormatter


if t.TYPE_CHECKING:
    DictStrStr = dict[str, str]
else:
    DictStrStr = dict


# TODO: Support jinja2 template, go template and possible other prompt template engine.
_default_formatter = PromptFormatter()


class PartialDict(DictStrStr):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


@attr.define(slots=True)
class PromptTemplate:
    template: str
    input_variables: t.Sequence[str]

    def to_str(self, __partial_dict__: PartialDict | None = None, **attrs: str) -> str:
        """Generate a prompt from the template and input variables."""
        if __partial_dict__:
            return _default_formatter.vformat(self.template, (), __partial_dict__)
        if not attrs:
            raise ValueError("Keyword arguments are required")
        if not all(k in attrs for k in self.input_variables):
            raise ValueError(f"Missing required input variables: {self.input_variables}")
        return _default_formatter.format(self.template, **attrs)

    @classmethod
    def from_template(cls, template: str) -> PromptTemplate:
        input_variables = _default_formatter.extract_template_variables(template)
        return cls(template=template, input_variables=input_variables)

    @classmethod
    def from_default(cls, model: str, /, **prompt_attrs: t.Any) -> PromptTemplate:
        template = openllm.utils.EnvVarMixin(model).module.DEFAULT_PROMPT_TEMPLATE
        if template is None:
            raise ValueError(f"Model {model} does not have a default prompt template.")
        if callable(template):
            template = template(**prompt_attrs)
        return cls.from_template(template)
