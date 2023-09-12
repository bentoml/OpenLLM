from __future__ import annotations
import string
import typing as t

class PromptFormatter(string.Formatter):
  """This PromptFormatter is largely based on langchain's implementation."""
  def vformat(self, format_string: str, args: t.Sequence[t.Any], kwargs: t.Mapping[str, t.Any]) -> t.Any:
    if len(args) > 0: raise ValueError('Positional arguments are not supported')
    return super().vformat(format_string, args, kwargs)

  def check_unused_args(self, used_args: set[int | str], args: t.Sequence[t.Any], kwargs: t.Mapping[str, t.Any]) -> None:
    extras = set(kwargs).difference(used_args)
    if extras: raise KeyError(f'Extra params passed: {extras}')

  def extract_template_variables(self, template: str) -> t.Sequence[str]:
    return [field[1] for field in self.parse(template) if field[1] is not None]

default_formatter = PromptFormatter()

def process_prompt(prompt: str, template: str | None = None, use_prompt_template: bool = True, **attrs: t.Any) -> str:
  # Currently, all default prompt will always have `instruction` key.
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
