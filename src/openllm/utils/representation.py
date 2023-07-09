# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import typing as t
from abc import abstractmethod

import attr
import orjson


if t.TYPE_CHECKING:
    ReprArgs: t.TypeAlias = t.Iterable[tuple[str | None, t.Any]]


class ReprMixin:
    """This class display possible representation of given class.

    It can be used for implementing __rich_pretty__ and __pretty__ methods in the future.
    Most subclass needs to implement a __repr_keys__ property.

    Based on the design from Pydantic.
    The __repr__ will display the json representation of the object for easier interaction.
    The __str__ will display either __attrs_repr__ or __repr_str__.
    """

    @property
    @abstractmethod
    def __repr_keys__(self) -> set[str]:
        """This can be overriden by base class using this mixin."""

    def __repr__(self) -> str:
        """The `__repr__` for any subclass of Mixin.

        It will print nicely the class name with each of the fields under '__repr_keys__' as kv JSON dict.
        """
        from . import bentoml_cattr

        serialized = {k: bentoml_cattr.unstructure(v) if attr.has(v) else v for k, v in self.__repr_args__()}
        return f"{self.__class__.__name__} {orjson.dumps(serialized, option=orjson.OPT_INDENT_2).decode()}"

    def __str__(self) -> str:
        """The string representation of the given Mixin subclass.

        It will contains all of the attributes from __repr_keys__
        """
        return self.__repr_str__(" ")

    def __repr_name__(self) -> str:
        """Name of the instance's class, used in __repr__."""
        return self.__class__.__name__

    def __repr_str__(self, join_str: str) -> str:
        """To be used with __str__."""
        return join_str.join(repr(v) if a is None else f"{a}={v!r}" for a, v in self.__repr_args__())

    def __repr_args__(self) -> ReprArgs:
        """This can also be overriden by base class using this mixin.

        By default it does a getattr of the current object from __repr_keys__.
        """
        return ((k, getattr(self, k)) for k in self.__repr_keys__)
