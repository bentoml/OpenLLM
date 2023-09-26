from __future__ import annotations
import typing as t

import attr

from .utils import PromptFormatter

default_formatter = PromptFormatter()

@attr.define(slots=True)
class PromptTemplate:
  _template: str | None
  _system_message: str | None

  def __init__(self, template: str | None = None, system_message: str | None = None) -> None:
    self._template = template
    self._system_message = system_message

  def _fill_in_variables(self, **kwargs: t.Any) -> str | None:
    if self._template is None: return self._template
    template_variables = default_formatter.extract_template_variables(self._template)
    prompt_variables = {key: '{' + key + '}' if key not in kwargs else (kwargs[key] or '') for key in template_variables}
    return self._template.format(**prompt_variables)

  @property
  def template(self) -> str | None:
    return self._fill_in_variables(system_message=self._system_message)

def process_prompt(prompt: str, template: PromptTemplate | str | None = None, use_prompt_template: bool = True, **attrs: t.Any) -> str:
  # Currently, all default prompt will always have `instruction` key.
  if isinstance(template, PromptTemplate): template = template.template
  if not use_prompt_template: return prompt
  elif template is None: raise ValueError("'template' can't be None while 'use_prompt_template=False'")
  template_variables = default_formatter.extract_template_variables(template)
  prompt_variables = {k: v for k, v in attrs.items() if k in template_variables}
  if 'instruction' in prompt_variables:
    raise RuntimeError("'instruction' should be passed as the first argument instead of kwargs when 'use_prompt_template=True'")
  try:
    return template.format(instruction=prompt, **prompt_variables)
  except KeyError as e:
    raise RuntimeError(
        f"Missing variable '{e.args[0]}' (required: {template_variables}) in the prompt template. Use 'use_prompt_template=False' to disable the default prompt template.") from None
