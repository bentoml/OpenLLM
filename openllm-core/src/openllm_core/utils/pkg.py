# NOTE: vendorred from bentoml._internal.utils.pkg
from __future__ import annotations
import importlib.metadata
import importlib.util
from importlib.metadata import PackageNotFoundError
from types import ModuleType
from typing import cast

from packaging.version import Version

__all__ = ['PackageNotFoundError', 'find_spec', 'get_pkg_version', 'pkg_version_info', 'source_locations']

get_pkg_version = importlib.metadata.version
find_spec = importlib.util.find_spec


def pkg_version_info(pkg_name: str | ModuleType) -> tuple[int, int, int]:
  if isinstance(pkg_name, ModuleType):
    pkg_name = pkg_name.__name__
  pkg_version = Version(get_pkg_version(pkg_name))
  return pkg_version.major, pkg_version.minor, pkg_version.micro


def source_locations(pkg: str) -> str | None:
  module = find_spec(pkg)
  if module is None:
    return None
  (module_path,) = module.submodule_search_locations  # type: ignore[misc]
  return cast(str, module_path)
