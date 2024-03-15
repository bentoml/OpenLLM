from __future__ import annotations
import functools, importlib, os, sys, typing as t
from enum import Enum
import attr, click, inflection, orjson, click_option_group as cog
from click import ParamType, shell_completion as sc, types as click_types
from .._typing_compat import overload, Unpack

if t.TYPE_CHECKING:
  from attr import _ValidatorType
  from pydantic import ConfigDict

T = t.TypeVar('T')
AnyCallable = t.Callable[..., t.Any]
FC = t.TypeVar('FC', bound=t.Union[AnyCallable, click.Command])

__all__ = [
  'CUDA',
  'FC',
  'BytesType',
  'EnumChoice',
  'Field',
  'JsonType',
  'LiteralChoice',
  'ModuleType',
  'allows_multiple',
  'attach_pydantic_model',
  'attrs_to_options',
  'is_container',
  'is_literal',
  'is_mapping',
  'is_typing',
  'parse_container_args',
  'parse_single_arg',
  'parse_type',
]


def __dir__() -> list[str]:
  return sorted(__all__)


def _error_callable(field: str):
  def fact():
    raise ValueError(f'"{field}" is required to be set from given attrs class.')

  return fact


@overload
def attach_pydantic_model(klass: t.Type[T], /, **config: Unpack[ConfigDict]) -> t.Type[T]: ...


@overload
def attach_pydantic_model(**config: Unpack[ConfigDict]) -> t.Callable[[t.Type[T]], t.Type[T]]: ...


def attach_pydantic_model(
  klass: t.Optional[t.Type[T]] = None, /, **config: Unpack[ConfigDict]
) -> t.Type[T] | t.Callable[[t.Type[T]], t.Type[T]]:
  # attach a cls.pydantic_model() -> pydantic.BaseModel compatible components.
  def _decorator(_cls: t.Type[T]) -> t.Type[T]:
    if not attr.has((_cls := attr.resolve_types(_cls))):
      raise TypeError('this decorator should only be used with attrs-compatible classes')

    from .pkg import pkg_version_info

    if (ver := pkg_version_info('pydantic')) < (2,):
      raise ImportError(f'Requires pydantic>=2.0, but found {".".join(map(str, ver))} instead.')

    try:
      from _bentoml_sdk.io_models import IODescriptor
    except ImportError:
      raise ImportError('Requires bentoml>=1.2 to be installed. Do "pip install -U "bentoml>=1.2""') from None

    import pydantic

    field_dict = {}
    for key, attrib in attr.fields_dict(_cls).items():
      attrib_type = resolve_attrib_types(attrib.type)
      if attrib.default is attr.NOTHING:
        field_dict[key] = (attrib_type, pydantic.Field(default_factory=_error_callable(key)))
      elif isinstance(attrib.default, attr.Factory):
        field_dict[key] = (attrib_type, pydantic.Field(default_factory=attrib.default.factory))
      else:
        field_dict[key] = (attrib_type, pydantic.Field(default=attrib.default))

    def create_dantic_class(cls):
      _klass = pydantic.create_model(
        cls.__name__ + 'Pydantic',
        __base__=(
          IODescriptor,
          pydantic.create_model(cls.__name__ + 'BaseModel', metadata_config=config, __module__=cls.__module__),
        ),
        __module__=cls.__module__,
        **field_dict,
      )
      _klass.media_type = 'application/json'
      return _klass

    setattr(_cls, 'pydantic_model', classmethod(create_dantic_class))
    setattr(_cls, '__openllm_attach_pydantic_model__', True)
    return _cls

  return _decorator if klass is None else _decorator(klass)


def resolve_attrib_types(typ_: t.Any) -> t.Any:
  if hasattr(typ_, 'pydantic_model'):
    return resolve_attrib_types(typ_.pydantic_model())
  if is_container(typ_):
    args = t.get_args(typ_)
    if len(args) == 0:
      return typ_
    # homogenous type
    if len(args) == 1 or (len(args) == 2 and args[1] is Ellipsis):
      if hasattr((single := args[0]), 'pydantic_model'):
        single = single.pydantic_model()
      return typ_.copy_with(single)
    return tuple(resolve_attrib_types(it) for it in args)
  return typ_


def attrs_to_options(
  name: str,
  field: attr.Attribute[t.Any],
  model_name: str,
  typ: t.Any = None,
  suffix_generation: bool = False,
  suffix_sampling: bool = False,
) -> t.Callable[[FC], FC]:
  # TODO: support parsing nested attrs class and Union
  envvar = field.metadata['env']
  dasherized = inflection.dasherize(name)
  underscored = inflection.underscore(name)

  if typ in (None, attr.NOTHING):
    typ = field.type
    if typ is None:
      raise RuntimeError(f'Failed to parse type for {name}')

  full_option_name = f'--{dasherized}'
  if field.type is bool:
    full_option_name += f'/--no-{dasherized}'
  if suffix_generation:
    identifier = f'{model_name}_generation_{underscored}'
  elif suffix_sampling:
    identifier = f'{model_name}_sampling_{underscored}'
  else:
    identifier = f'{model_name}_{underscored}'

  return cog.optgroup.option(
    identifier,
    full_option_name,
    type=parse_type(typ),
    required=field.default is attr.NOTHING,
    default=field.default if field.default not in (attr.NOTHING, None) else None,
    show_default=True,
    multiple=allows_multiple(typ) if typ else False,
    help=field.metadata.get('description', '(No description provided)'),
    show_envvar=True,
    envvar=envvar,
  )


def env_converter(value: t.Any, env: str | None = None) -> t.Any:
  if env is not None:
    value = os.environ.get(env, value)
    if value is not None and isinstance(value, str):
      try:
        return orjson.loads(value.lower())
      except orjson.JSONDecodeError as err:
        raise RuntimeError(f"Failed to parse ({value!r}) from '{env}': {err}") from None
  return value


def Field(
  default: t.Any = None,
  *,
  ge: int | float | None = None,
  le: int | float | None = None,
  validator: _ValidatorType[t.Any] | None = None,
  description: str | None = None,
  env: str | None = None,
  auto_default: bool = False,
  use_default_converter: bool = True,
  **attrs: t.Any,
) -> t.Any:
  """A decorator that extends attr.field with additional arguments, which provides the same interface as pydantic's Field.

  By default, if both validator and ge are provided, then then ge will be
  piped into first, then all of the other validator will be run afterwards.

  Args:
  default: The default value for ``dantic.Field``. Defaults to ``None``.
  ge: Greater than or equal to. Defaults to None.
  le: Less than or equal to. Defaults to None.
  validator: Optional attrs-compatible validators type. Default to None
  description: the documentation for the field. Defaults to None.
  env: the environment variable to read from. Defaults to None.
  auto_default: a bool indicating whether to use the default value as the environment.
  Defaults to False. If set to True, the behaviour of this Field will also depends
  on kw_only. If kw_only=True, the this field will become 'Required' and the default
  value is omitted. If kw_only=False, then the default value will be used as before.
  use_default_converter: a bool indicating whether to use the default converter. Defaults
  to True. If set to False, then the default converter will not be used.
  The default converter converts a given value from the environment variable
  for this given Field.
  **attrs: The rest of the arguments are passed to attr.field
  """
  metadata = attrs.pop('metadata', {})
  if description is None:
    description = '(No description provided)'
  metadata['description'] = description
  if env is not None:
    metadata['env'] = env
  piped: list[_ValidatorType[t.Any]] = []

  converter = attrs.pop('converter', None)
  if use_default_converter:
    converter = functools.partial(env_converter, env=env)

  if ge is not None:
    piped.append(attr.validators.ge(ge))
  if le is not None:
    piped.append(attr.validators.le(le))
  if validator is not None:
    piped.append(validator)

  if len(piped) == 0:
    _validator = None
  elif len(piped) == 1:
    _validator = piped[0]
  else:
    _validator = attr.validators.and_(*piped)

  factory = attrs.pop('factory', None)
  if factory is not None and default is not None:
    raise RuntimeError("'factory' and 'default' are mutually exclusive.")
  # NOTE: the behaviour of this is we will respect factory over the default
  if factory is not None:
    attrs['factory'] = factory
  else:
    attrs['default'] = default

  kw_only = attrs.pop('kw_only', False)
  if auto_default and kw_only:
    attrs.pop('default')

  return attr.field(metadata=metadata, validator=_validator, converter=converter, **attrs)


def parse_type(field_type: t.Any) -> ParamType | tuple[ParamType, ...]:
  """Transforms the pydantic field's type into a click-compatible type.

  Args:
  field_type: pydantic field type

  Returns:
  ParamType: click type equivalent
  """
  from . import lenient_issubclass

  if t.get_origin(field_type) is t.Union:
    raise NotImplementedError('Unions are not supported')
  # enumeration strings or other Enum derivatives
  if lenient_issubclass(field_type, Enum):
    return EnumChoice(enum=field_type, case_sensitive=True)
  # literals are enum-like with way less functionality
  if is_literal(field_type):
    return LiteralChoice(value=field_type, case_sensitive=True)
  # modules, classes, functions
  if is_typing(field_type):
    return ModuleType()
  # entire dictionaries:
  # using a Dict, convert in advance
  if is_mapping(field_type):
    return JsonType()
  # list, List[p], Tuple[p], Set[p] and so on
  if is_container(field_type):
    return parse_container_args(field_type)
  # bytes are not natively supported by click
  if lenient_issubclass(field_type, bytes):
    return BytesType()
  # return the current type: it should be a primitive
  return field_type


def is_typing(field_type: type) -> bool:
  """Checks whether the current type is a module-like type.

  Args:
  field_type: pydantic field type

  Returns:
  bool: true if the type is itself a type
  """
  raw = t.get_origin(field_type)
  if raw is None:
    return False
  if raw is type or raw is t.Type:
    return True
  return False


def is_literal(field_type: type) -> bool:
  """Checks whether the given field type is a Literal type or not.

  Literals are weird: isinstance and subclass do not work, so you compare
  the origin with the Literal declaration itself.

  Args:
  field_type: current pydantic type

  Returns:
  bool: true if Literal type, false otherwise
  """
  origin = t.get_origin(field_type)
  return origin is not None and origin is t.Literal


class ModuleType(ParamType):
  name = 'module'

  def _import_object(self, value: str) -> t.Any:
    module_name, class_name = value.rsplit('.', maxsplit=1)
    if not all(s.isidentifier() for s in module_name.split('.')):
      raise ValueError(f"'{value}' is not a valid module name")
    if not class_name.isidentifier():
      raise ValueError(f"Variable '{class_name}' is not a valid identifier")

    module = importlib.import_module(module_name)
    if class_name:
      try:
        return getattr(module, class_name)
      except AttributeError:
        raise ImportError(f"Module '{module_name}' does not define a '{class_name}' variable.") from None

  def convert(self, value: str | t.Any, param: click.Parameter | None, ctx: click.Context | None) -> t.Any:
    try:
      if isinstance(value, str):
        return self._import_object(value)
      return value
    except Exception as exc:
      self.fail(f"'{value}' is not a valid object ({type(exc)}: {exc!s})", param, ctx)


class EnumChoice(click.Choice):
  name = 'enum'

  def __init__(self, enum: Enum, case_sensitive: bool = False):
    """Enum type support for click that extends ``click.Choice``.

    Args:
    enum: Given enum
    case_sensitive: Whether this choice should be case case_sensitive.
    """
    self.mapping = enum
    self.internal_type = type(enum)
    choices: list[t.Any] = [e.name for e in enum.__class__]
    super().__init__(choices, case_sensitive)

  def convert(self, value: t.Any, param: click.Parameter | None, ctx: click.Context | None) -> Enum:
    if isinstance(value, self.internal_type):
      return value
    result = super().convert(value, param, ctx)
    if isinstance(result, str):
      result = self.internal_type[result]
    return result


class LiteralChoice(EnumChoice):
  name = 'literal'

  def __init__(self, value: t.Any, case_sensitive: bool = False):
    """Literal support for click."""
    # expect every literal value to belong to the same primitive type
    values = list(value.__args__)
    item_type = type(values[0])
    if not all(isinstance(v, item_type) for v in values):
      raise ValueError(f'Field {value} contains items of different types.')
    _mapping = {str(v): v for v in values}
    super(EnumChoice, self).__init__(list(_mapping), case_sensitive)
    self.internal_type = item_type


def allows_multiple(field_type: type[t.Any]) -> bool:
  """Checks whether the current type allows for multiple arguments to be provided as input or not.

  For containers, it exploits click's support for lists and such to use the same option multiple times
  to create a complex object: `python run.py --subsets train --subsets test`
  # becomes `subsets: ["train", "test"]`.

  Args:
  field_type: pydantic type.

  Returns:
  bool: true if it's a composite field (lists, containers and so on), false otherwise
  """
  # Early out for mappings, since it's better to deal with them using strings.
  if is_mapping(field_type):
    return False
  # Activate multiple option for (simple) container types
  if is_container(field_type):
    args = parse_container_args(field_type)
    # A non-composite type has a single argument, such as 'List[int]'
    # A composite type has a tuple of arguments, like 'Tuple[str, int, int]'.
    # For the moment, only non-composite types are allowed.
    return not isinstance(args, tuple)
  return False


def is_mapping(field_type: type) -> bool:
  """Checks whether this field represents a dictionary or JSON object.

  Args:
  field_type (type): pydantic type

  Returns:
  bool: true when the field is a dict-like object, false otherwise.
  """
  # Early out for standard containers.
  from . import lenient_issubclass

  if lenient_issubclass(field_type, t.Mapping):
    return True
  # for everything else or when the typing is more complex, check its origin
  origin = t.get_origin(field_type)
  if origin is None:
    return False
  return lenient_issubclass(origin, t.Mapping)


def is_container(field_type: type) -> bool:
  """Checks whether the current type is a container type ('contains' other types), like lists and tuples.

  Args:
  field_type: pydantic field type

  Returns:
  bool: true if a container, false otherwise
  """
  # do not consider strings or byte arrays as containers
  if field_type in (str, bytes):
    return False
  # Early out for standard containers: list, tuple, range
  from . import lenient_issubclass

  if lenient_issubclass(field_type, t.Container):
    return True
  origin = t.get_origin(field_type)
  # Early out for non-typing objects
  if origin is None:
    return False
  return lenient_issubclass(origin, t.Container)


def parse_container_args(field_type: type[t.Any]) -> ParamType | tuple[ParamType, ...]:
  """Parses the arguments inside a container type (lists, tuples and so on).

  Args:
  field_type: pydantic field type

  Returns:
  ParamType | tuple[ParamType]: single click-compatible type or a tuple
  """
  if not is_container(field_type):
    raise ValueError('Field type is not a container type.')
  args = t.get_args(field_type)
  # Early out for untyped containers: standard lists, tuples, List[Any]
  # Use strings when the type is unknown, avoid click's type guessing
  if len(args) == 0:
    return click_types.convert_type(str)
  # Early out for homogenous containers: Tuple[int], List[str]
  # or homogenous tuples of indefinite length: Tuple[int, ...]
  if len(args) == 1 or (len(args) == 2 and args[1] is Ellipsis):
    return parse_single_arg(args[0])
  # Then deal with fixed-length containers: Tuple[str, int, int]
  return tuple(parse_single_arg(arg) for arg in args)


def parse_single_arg(arg: type) -> ParamType:
  """Returns the click-compatible type for container origin types.

  In this case, returns string when it's not inferrable, a JSON for mappings
  and the original type itself in every other case (ints, floats and so on).
  Bytes is a special case, not natively handled by click.

  Args:
  arg (type): single argument

  Returns:
  ParamType: click-compatible type
  """
  from . import lenient_issubclass

  # When we don't know the type, we choose 'str'
  if arg is t.Any:
    return click_types.convert_type(str)
  # For containers and nested models, we use JSON
  if is_container(arg):
    return JsonType()
  if lenient_issubclass(arg, bytes):
    return BytesType()
  return click_types.convert_type(arg)


class BytesType(ParamType):
  name = 'bytes'

  def convert(self, value: t.Any, param: click.Parameter | None, ctx: click.Context | None) -> t.Any:
    if isinstance(value, bytes):
      return value
    try:
      return str.encode(value)
    except Exception as exc:
      self.fail(f"'{value}' is not a valid string ({exc!s})", param, ctx)


CYGWIN = sys.platform.startswith('cygwin')
WIN = sys.platform.startswith('win')
if sys.platform.startswith('win') and WIN:

  def _get_argv_encoding() -> str:
    import locale

    return locale.getpreferredencoding()
else:

  def _get_argv_encoding() -> str:
    return getattr(sys.stdin, 'encoding', None) or sys.getfilesystemencoding()


class CudaValueType(ParamType):
  name = 'cuda'
  envvar_list_splitter = ','
  is_composite = True

  def split_envvar_value(self, rv: str) -> t.Sequence[str]:
    var = tuple(i for i in rv.split(self.envvar_list_splitter))
    if '-1' in var:
      return var[: var.index('-1')]
    return var

  def shell_complete(self, ctx: click.Context, param: click.Parameter, incomplete: str) -> list[sc.CompletionItem]:
    """Return a list of :class:`~click.shell_completion.CompletionItem` objects for the incomplete value.

    Most types do not provide completions, but some do, and this allows custom types to provide custom completions as well.

    Args:
    ctx: Invocation context for this command.
    param: The parameter that is requesting completion.
    incomplete: Value being completed. May be empty.
    """
    from openllm.utils import available_devices

    mapping = incomplete.split(self.envvar_list_splitter) if incomplete else available_devices()
    return [sc.CompletionItem(str(i), help=f'CUDA device index {i}') for i in mapping]

  def convert(self, value: t.Any, param: click.Parameter | None, ctx: click.Context | None) -> t.Any:
    typ = click_types.convert_type(str)
    if isinstance(value, bytes):
      enc = _get_argv_encoding()
      try:
        value = value.decode(enc)
      except UnicodeError:
        fs_enc = sys.getfilesystemencoding()
        if fs_enc != enc:
          try:
            value = value.decode(fs_enc)
          except UnicodeError:
            value = value.decode('utf-8', 'replace')
        else:
          value = value.decode('utf-8', 'replace')
    return tuple(typ(x, param, ctx) for x in value.split(','))

  def __repr__(self) -> str:
    return 'STRING'


CUDA = CudaValueType()


class JsonType(ParamType):
  name = 'json'

  def __init__(self, should_load: bool = True) -> None:
    """Support JSON type for click.ParamType.

    Args:
    should_load: Whether to load the JSON. Default to True. If False, the value won't be converted.
    """
    super().__init__()
    self.should_load = should_load

  def convert(self, value: t.Any, param: click.Parameter | None, ctx: click.Context | None) -> t.Any:
    if isinstance(value, dict) or not self.should_load:
      return value
    try:
      return orjson.loads(value)
    except orjson.JSONDecodeError as exc:
      self.fail(f"'{value}' is not a valid JSON string ({exc!s})", param, ctx)
