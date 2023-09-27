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
