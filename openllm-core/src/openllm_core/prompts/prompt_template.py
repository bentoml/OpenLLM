from __future__ import annotations
import typing as t

from ._prompt import default_formatter

class PromptTemplate:
  def __init__(self, template: str | None = None, system_message: str | None = None) -> None:
    self._template = template
    self.system_message = system_message

  def _fill_in_variables(self, **kwargs: t.Any) -> str | None:
    if self._template is None: return self._template
    template_variables = default_formatter.extract_template_variables(self._template)
    prompt_variables = {key: '{' + key + '}' if key not in kwargs else kwargs[key] for key in template_variables}
    return self._template.format(**prompt_variables)

  @property
  def template(self) -> str | None:
    return self._fill_in_variables(system_message=self.system_message)

  @template.setter
  def template(self, t: str) -> None:
    self._template = t
