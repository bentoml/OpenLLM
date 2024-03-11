from __future__ import annotations
import functools
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import itertools
import logging
import os
import sys
import time
import types
import typing as t
import warnings

import attr

import openllm_core

__all__ = ['LazyLoader', 'LazyModule', 'VersionInfo']

logger = logging.getLogger(__name__)


class LazyLoader(types.ModuleType):
  """
  LazyLoader module borrowed from Tensorflow https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/util/lazy_loader.py with a addition of "module caching". This will throw an exception if module cannot be imported.

  Lazily import a module, mainly to avoid pulling in large dependencies. `contrib`, and `ffmpeg` are examples of modules that are large and not always needed, and this allows them to only be loaded when they are used.
  """

  def __init__(
    self,
    local_name: str,
    parent_module_globals: dict[str, t.Any],
    name: str,
    warning: str | None = None,
    exc_msg: str | None = None,
    exc: type[BaseException] = openllm_core.exceptions.MissingDependencyError,
  ):
    self._local_name = local_name
    self._parent_module_globals = parent_module_globals
    self._warning = warning
    self._exc_msg = exc_msg
    self._exc = exc
    self._module: types.ModuleType | None = None

    super().__init__(str(name))

  def _load(self) -> types.ModuleType:
    """Load the module and insert it into the parent's globals."""
    # Import the target module and insert it into the parent's namespace
    try:
      module = importlib.import_module(self.__name__)
      self._parent_module_globals[self._local_name] = module
      # The additional add to sys.modules ensures library is actually loaded.
      sys.modules[self._local_name] = module
    except ModuleNotFoundError as err:
      raise self._exc(f'{self._exc_msg} (reason: {err})') from None

    # Emit a warning if one was specified
    if self._warning:
      logger.warning(self._warning)
      # Make sure to only warn once.
      self._warning = None

    # Update this object's dict so that if someone keeps a reference to the
    #   LazyLoader, lookups are efficient (__getattr__ is only called on lookups
    #   that fail).
    self.__dict__.update(module.__dict__)
    return module

  def __getattr__(self, item: t.Any) -> t.Any:
    if self._module is None:
      self._module = self._load()
    return getattr(self._module, item)

  def __dir__(self) -> t.List[str]:
    if self._module is None:
      self._module = self._load()
    return dir(self._module)


# vendorred from attrs
@functools.total_ordering
@attr.attrs(eq=False, order=False, slots=True, frozen=True, repr=False)
class VersionInfo:
  major: int = attr.field()
  minor: int = attr.field()
  micro: int = attr.field()
  releaselevel: str = attr.field()

  @classmethod
  def from_package(cls, package: str) -> VersionInfo:
    return cls.from_version_string(importlib.metadata.version(package))

  @classmethod
  def from_version_string(cls, s: str) -> VersionInfo:
    v = s.split('.')
    if len(v) == 3:
      v.append('final')
    return cls(major=int(v[0]), minor=int(v[1]), micro=int(v[2]), releaselevel=v[3])

  def _ensure_tuple(self, other: VersionInfo) -> tuple[tuple[int, int, int, str], tuple[int, int, int, str]]:
    cmp = attr.astuple(other) if self.__class__ is other.__class__ else other
    if not isinstance(cmp, tuple):
      raise NotImplementedError
    if not (1 <= len(cmp) <= 4):
      raise NotImplementedError
    return t.cast(t.Tuple[int, int, int, str], attr.astuple(self)[: len(cmp)]), t.cast(
      t.Tuple[int, int, int, str], cmp
    )

  def __eq__(self, other: t.Any) -> bool:
    try:
      us, them = self._ensure_tuple(other)
    except NotImplementedError:
      return NotImplemented
    return us == them

  def __lt__(self, other: t.Any) -> bool:
    try:
      us, them = self._ensure_tuple(other)
    except NotImplementedError:
      return NotImplemented
    # Since alphabetically "dev0" < "final" < "post1" < "post2", we don't have to do anything special with releaselevel for now.
    return us < them

  def __repr__(self) -> str:
    return '{0}.{1}.{2}'.format(*attr.astuple(self)[:3])


_sentinel, _reserved_namespace = object(), {'__openllm_migration__'}


class LazyModule(types.ModuleType):
  # Very heavily inspired by optuna.integration._IntegrationModule: https://github.com/optuna/optuna/blob/master/optuna/integration/__init__.py
  def __init__(
    self,
    name: str,
    module_file: str,
    import_structure: dict[str, list[str]],
    module_spec: importlib.machinery.ModuleSpec | None = None,
    doc: str | None = None,
    extra_objects: dict[str, t.Any] | None = None,
  ):
    """Lazily load this module as an object.

    It does instantiate a __all__ and __dir__ for IDE support

    Args:
      name: module name
      module_file: the given file. Often default to 'globals()['__file__']'
      import_structure: A dictionary of module and its corresponding attributes that can be loaded from given 'module'
      module_spec: __spec__ of the lazily loaded module
      doc: Optional docstring for this module.
      extra_objects: Any additional objects that this module can also be accessed. Useful for additional metadata as well as any locals() functions
    """
    super().__init__(name)
    self._modules = set(import_structure.keys())
    self._class_to_module: dict[str, str] = {}
    _extra_objects = {} if extra_objects is None else extra_objects
    for key, values in import_structure.items():
      for value in values:
        self._class_to_module[value] = key
    # Needed for autocompletion in an IDE
    self.__all__: list[str] = list(import_structure.keys()) + list(itertools.chain(*import_structure.values()))
    self.__file__ = module_file
    self.__spec__ = module_spec or importlib.util.find_spec(name)
    self.__path__ = [os.path.dirname(module_file)]
    self.__doc__ = doc
    self._name = name
    self._objects = _extra_objects
    self._import_structure = import_structure

  def __dir__(self) -> list[str]:
    result = t.cast('list[str]', super().__dir__())
    # The elements of self.__all__ that are submodules may or may not be in the dir already, depending on whether
    # they have been accessed or not. So we only add the elements of self.__all__ that are not already in the dir.
    return result + [i for i in self.__all__ if i not in result]

  def __getattr__(self, name: str) -> t.Any:
    """Equivocal __getattr__ implementation.

    It checks from _objects > _modules and does it recursively.

    It also contains a special case for all of the metadata information, such as __version__ and __version_info__.
    """
    if name in _reserved_namespace:
      raise openllm_core.exceptions.ForbiddenAttributeError(
        f"'{name}' is a reserved namespace for {self._name} and should not be access nor modified."
      )
    dunder_to_metadata = {
      '__title__': 'Name',
      '__copyright__': '',
      '__version__': 'version',
      '__version_info__': 'version',
      '__description__': 'summary',
      '__uri__': '',
      '__url__': '',
      '__author__': '',
      '__email__': '',
      '__license__': 'license',
      '__homepage__': '',
    }
    if name in dunder_to_metadata:
      if name not in {'__version_info__', '__copyright__', '__version__'}:
        warnings.warn(
          f"Accessing '{self._name}.{name}' is deprecated. Please consider using 'importlib.metadata' directly to query for openllm packaging metadata.",
          DeprecationWarning,
          stacklevel=2,
        )
      meta = importlib.metadata.metadata('openllm')
      project_url = dict(url.split(', ') for url in t.cast(t.List[str], meta.get_all('Project-URL')))
      if name == '__license__':
        return 'Apache-2.0'
      elif name == '__copyright__':
        return f"Copyright (c) 2023-{time.strftime('%Y')}, Aaron Pham et al."
      elif name in ('__uri__', '__url__'):
        return project_url['GitHub']
      elif name == '__homepage__':
        return project_url['Homepage']
      elif name == '__version_info__':
        return VersionInfo.from_version_string(meta['version'])  # similar to how attrs handle __version_info__
      elif name == '__author__':
        return meta['Author-email'].rsplit(' ', 1)[0]
      elif name == '__email__':
        return meta['Author-email'].rsplit('<', 1)[1][:-1]
      return meta[dunder_to_metadata[name]]
    if '__openllm_migration__' in self._objects:
      cur_value = self._objects['__openllm_migration__'].get(name, _sentinel)
      if cur_value is not _sentinel:
        warnings.warn(
          f"'{name}' is deprecated and will be removed in future version. Make sure to use '{cur_value}' instead",
          DeprecationWarning,
          stacklevel=3,
        )
        return getattr(self, cur_value)
    if name in self._objects:
      return self._objects.__getitem__(name)
    if name in self._modules:
      value = self._get_module(name)
    elif name in self._class_to_module.keys():
      value = getattr(self._get_module(self._class_to_module.__getitem__(name)), name)
    # special branch for openllm_core
    elif hasattr(openllm_core, name):
      return getattr(openllm_core, name)
    else:
      raise AttributeError(f'module {self.__name__} has no attribute {name}')
    setattr(self, name, value)
    return value

  def _get_module(self, module_name: str) -> types.ModuleType:
    try:
      return importlib.import_module('.' + module_name, self.__name__)
    except Exception as e:
      raise RuntimeError(
        f'Failed to import {self.__name__}.{module_name} because of the following error (look up to see its traceback):\n{e}'
      ) from e

  # make sure this module is picklable
  def __reduce__(self) -> tuple[type[LazyModule], tuple[str, str | None, dict[str, list[str]]]]:
    return (self.__class__, (self._name, self.__file__, self._import_structure))
