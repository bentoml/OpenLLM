from __future__ import annotations
import functools
import inspect
import linecache
import logging
import types
import typing as t
from operator import itemgetter

import orjson
if t.TYPE_CHECKING:
  import openllm_core
  from openllm_core._typing_compat import AnyCallable, DictStrAny, ListStr, LiteralString
  PartialAny = functools.partial[t.Any]

_T = t.TypeVar('_T', bound=t.Callable[..., t.Any])
logger = logging.getLogger(__name__)

# sentinel object for unequivocal object() getattr
_sentinel = object()

def has_own_attribute(cls: type[t.Any], attrib_name: t.Any) -> bool:
  """Check whether *cls* defines *attrib_name* (and doesn't just inherit it)."""
  attr = getattr(cls, attrib_name, _sentinel)
  if attr is _sentinel: return False
  for base_cls in cls.__mro__[1:]:
    a = getattr(base_cls, attrib_name, None)
    if attr is a: return False
  return True

def get_annotations(cls: type[t.Any]) -> DictStrAny:
  if has_own_attribute(cls, '__annotations__'): return cls.__annotations__
  return t.cast('DictStrAny', {})

def is_class_var(annot: str | t.Any) -> bool:
  annot = str(annot)
  # Annotation can be quoted.
  if annot.startswith(("'", '"')) and annot.endswith(("'", '"')): annot = annot[1:-1]
  return annot.startswith(('typing.ClassVar', 't.ClassVar', 'ClassVar', 'typing_extensions.ClassVar',))

def add_method_dunders(cls: type[t.Any], method_or_cls: _T, _overwrite_doc: str | None = None) -> _T:
  try:
    method_or_cls.__module__ = cls.__module__
  except AttributeError:
    pass
  try:
    method_or_cls.__qualname__ = f'{cls.__qualname__}.{method_or_cls.__name__}'
  except AttributeError:
    pass
  try:
    method_or_cls.__doc__ = _overwrite_doc or 'Generated by ``openllm.LLMConfig`` for class ' f'{cls.__qualname__}.'
  except AttributeError:
    pass
  return method_or_cls

def _compile_and_eval(script: str, globs: DictStrAny, locs: t.Any = None, filename: str = '') -> None:
  eval(compile(script, filename, 'exec'), globs, locs)

def _make_method(name: str, script: str, filename: str, globs: DictStrAny) -> AnyCallable:
  locs: DictStrAny = {}
  # In order of debuggers like PDB being able to step through the code, we add a fake linecache entry.
  count = 1
  base_filename = filename
  while True:
    linecache_tuple = (len(script), None, script.splitlines(True), filename)
    old_val = linecache.cache.setdefault(filename, linecache_tuple)
    if old_val == linecache_tuple: break
    else:
      filename = f'{base_filename[:-1]}-{count}>'
      count += 1
  _compile_and_eval(script, globs, locs, filename)
  return locs[name]

def make_attr_tuple_class(cls_name: str, attr_names: t.Sequence[str]) -> type[t.Any]:
  '''Create a tuple subclass to hold class attributes.

  The subclass is a bare tuple with properties for names.

  class MyClassAttributes(tuple):
  __slots__ = ()
  x = property(itemgetter(0))
  '''
  from . import SHOW_CODEGEN

  attr_class_name = f'{cls_name}Attributes'
  attr_class_template = [f'class {attr_class_name}(tuple):', '    __slots__ = ()',]
  if attr_names:
    for i, attr_name in enumerate(attr_names):
      attr_class_template.append(f'    {attr_name} = _attrs_property(_attrs_itemgetter({i}))')
  else:
    attr_class_template.append('    pass')
  globs: DictStrAny = {'_attrs_itemgetter': itemgetter, '_attrs_property': property}
  if SHOW_CODEGEN: logger.info('Generated class for %s:\n\n%s', attr_class_name, '\n'.join(attr_class_template))
  _compile_and_eval('\n'.join(attr_class_template), globs)
  return globs[attr_class_name]

def generate_unique_filename(cls: type[t.Any], func_name: str) -> str:
  return f"<{cls.__name__} generated {func_name} {cls.__module__}.{getattr(cls, '__qualname__', cls.__name__)}>"

def generate_function(
    typ: type[t.Any], func_name: str, lines: list[str] | None, args: tuple[str, ...] | None, globs: dict[str, t.Any], annotations: dict[str, t.Any] | None = None
) -> AnyCallable:
  from openllm_core.utils import SHOW_CODEGEN
  script = 'def %s(%s):\n    %s\n' % (func_name, ', '.join(args) if args is not None else '', '\n    '.join(lines) if lines else 'pass')
  meth = _make_method(func_name, script, generate_unique_filename(typ, func_name), globs)
  if annotations: meth.__annotations__ = annotations
  if SHOW_CODEGEN: logger.info('Generated script for %s:\n\n%s', typ, script)
  return meth

def make_env_transformer(
    cls: type[openllm_core.LLMConfig], model_name: str, suffix: LiteralString | None = None, default_callback: t.Callable[[str, t.Any], t.Any] | None = None, globs: DictStrAny | None = None,
) -> AnyCallable:
  from openllm_core.utils import dantic, field_env_key

  def identity(_: str, x_value: t.Any) -> t.Any:
    return x_value

  default_callback = identity if default_callback is None else default_callback
  globs = {} if globs is None else globs
  globs.update({'__populate_env': dantic.env_converter, '__default_callback': default_callback, '__field_env': field_env_key, '__suffix': suffix or '', '__model_name': model_name,})
  lines: ListStr = [
      '__env = lambda field_name: __field_env(__model_name, field_name, __suffix)',
      'return [',
      '    f.evolve(',
      '        default=__populate_env(__default_callback(f.name, f.default), __env(f.name)),',
      '        metadata={',
      "            'env': f.metadata.get('env', __env(f.name)),",
      "            'description': f.metadata.get('description', '(not provided)'),",
      '        },',
      '    )',
      '    for f in fields',
      ']'
  ]
  fields_ann = 'list[attr.Attribute[t.Any]]'
  return generate_function(cls, '__auto_env', lines, args=('_', 'fields'), globs=globs, annotations={'_': 'type[LLMConfig]', 'fields': fields_ann, 'return': fields_ann})

def gen_sdk(func: _T, name: str | None = None, **attrs: t.Any) -> _T:
  '''Enhance sdk with nice repr that plays well with your brain.'''
  from openllm_core.utils import ReprMixin
  if name is None: name = func.__name__.strip('_')
  _signatures = inspect.signature(func).parameters

  def _repr(self: ReprMixin) -> str:
    return f'<generated function {name} {orjson.dumps(dict(self.__repr_args__()), option=orjson.OPT_NON_STR_KEYS | orjson.OPT_INDENT_2).decode()}>'

  def _repr_args(self: ReprMixin) -> t.Iterator[t.Tuple[str, t.Any]]:
    return ((k, _signatures[k].annotation) for k in self.__repr_keys__)

  if func.__doc__ is None: doc = f'Generated SDK for {func.__name__}'
  else: doc = func.__doc__
  return t.cast(
      _T,
      functools.update_wrapper(
          types.new_class(
              name, (t.cast('PartialAny', functools.partial), ReprMixin),
              exec_body=lambda ns: ns.update({
                  '__repr_keys__': property(lambda _: [i for i in _signatures.keys() if not i.startswith('_')]),
                  '__repr_args__': _repr_args,
                  '__repr__': _repr,
                  '__doc__': inspect.cleandoc(doc),
                  '__module__': 'openllm'
              }),
          )(func, **attrs),
          func,
      )
  )

__all__ = ['gen_sdk', 'make_attr_tuple_class', 'make_env_transformer', 'generate_unique_filename', 'generate_function']
