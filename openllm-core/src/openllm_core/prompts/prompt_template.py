from __future__ import annotations
import typing as t

import attr

from .utils import default_formatter

# equivocal setattr to save one lookup per assignment
_object_setattr = object.__setattr__

@attr.define(slots=True)
class PromptTemplate:
  template: str
  _input_variables: t.Sequence[str] = attr.field(init=False)

  def __attrs_post_init__(self) -> None:
    self._input_variables = default_formatter.extract_template_variables(self.template)

  def with_options(self, **attrs: t.Any) -> PromptTemplate:
    prompt_variables = {key: '{' + key + '}' if key not in attrs else attrs[key] for key in self._input_variables}
    o = attr.evolve(self, template=self.template.format(**prompt_variables))
    _object_setattr(o, '_input_variables', default_formatter.extract_template_variables(o.template))
    return o

  def to_string(self) -> str:
    return self.template

  def format(self, **attrs: t.Any) -> str:
    prompt_variables = {k: v for k, v in attrs.items() if k in self._input_variables}
    try:
      return self.template.format(**prompt_variables)
    except KeyError as e:
      raise RuntimeError(f"Missing variable '{e.args[0]}' (required: {self._input_variables}) in the prompt template.") from None

# TODO: remove process_prompt after refactor config for all models
def process_prompt(prompt: str, template: PromptTemplate | str | None = None, use_prompt_template: bool = True, **attrs: t.Any) -> str:
  # Currently, all default prompt will always have `instruction` key.
  if not use_prompt_template: return prompt
  elif template is None: raise ValueError("'template' can't be None while 'use_prompt_template=False'")
  if isinstance(template, PromptTemplate): template = template.to_string()
  template_variables = default_formatter.extract_template_variables(template)
  prompt_variables = {k: v for k, v in attrs.items() if k in template_variables}
  if 'instruction' in prompt_variables:
    raise RuntimeError("'instruction' should be passed as the first argument instead of kwargs when 'use_prompt_template=True'")
  try:
    return template.format(instruction=prompt, **prompt_variables)
  except KeyError as e:
    raise RuntimeError(
        f"Missing variable '{e.args[0]}' (required: {template_variables}) in the prompt template. Use 'use_prompt_template=False' to disable the default prompt template.") from None
