from __future__ import annotations

import openllm, logging, typing as t
from openllm_core._typing_compat import TypedDict, Required

if t.TYPE_CHECKING:
  from pydantic import BaseModel

logger = logging.getLogger(__name__)


def create_typeddict_from_model(model: type[BaseModel]) -> dict[str, t.Any]:
  globs = {'TypedDict': TypedDict, 'typing': t, 't': t, 'Required': Required}
  name = f'{model.__name__}TypedDict'
  lines = [f'class {name}(TypedDict, total=False):']
  for key, field_info in model.model_fields.items():
    if field_info.is_required():
      typ = f'Required[{model.__annotations__[key]}]'
    else:
      typ = model.__annotations__[key]
    lines.append(f'  {key}: {typ}')

  script = '\n'.join(lines)

  if openllm.utils.DEBUG:
    logger.info('Generated class:\n%s', script)

  eval(compile(script, 'name', 'exec'), globs)
  return globs[name]
