from __future__ import annotations
import typing as t

import attr

from .utils import default_formatter

@attr.define(slots=True)
class PromptTemplate:
  prompt_template: str
  input_variables: t.Mapping[str, str]
  prepared_template: str = attr.field(init=False)

  def __attrs_post_init__(self) -> None:
    template_variables = default_formatter.extract_template_variables(self.prompt_template)
    prompt_variables = {key: '{' + key + '}' if key not in self.input_variables else (self.input_variables[key] or '') for key in template_variables}
    self.prepared_template = self.prompt_template.format(**prompt_variables)

def process_prompt(prompt: str, template: PromptTemplate | str | None = None, use_prompt_template: bool = True, **attrs: t.Any) -> str:
  # Currently, all default prompt will always have `instruction` key.
  if not use_prompt_template: return prompt
  elif template is None: raise ValueError("'template' can't be None while 'use_prompt_template=False'")
  if isinstance(template, PromptTemplate): template = template.prepared_template
  template_variables = default_formatter.extract_template_variables(template)
  prompt_variables = {k: v for k, v in attrs.items() if k in template_variables}
  if 'instruction' in prompt_variables:
    raise RuntimeError("'instruction' should be passed as the first argument instead of kwargs when 'use_prompt_template=True'")
  try:
    return template.format(instruction=prompt, **prompt_variables)
  except KeyError as e:
    raise RuntimeError(
        f"Missing variable '{e.args[0]}' (required: {template_variables}) in the prompt template. Use 'use_prompt_template=False' to disable the default prompt template.") from None
