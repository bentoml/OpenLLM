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

import logging
import os
import string
import typing as t
from pathlib import Path

import orjson


if t.TYPE_CHECKING:
    from fs.base import FS

    import openllm

    DictStrAny = dict[str, t.Any]
    ListStr = list[str]

    from attr import _make_method
else:
    # NOTE: Using internal API from attr here, since we are actually
    # allowing subclass of openllm.LLMConfig to become 'attrs'-ish
    from attr._make import _make_method

    DictStrAny = dict
    ListStr = list

_T = t.TypeVar("_T", bound=t.Callable[..., t.Any])

logger = logging.getLogger(__name__)

OPENLLM_MODEL_NAME = "# openllm: model name"
OPENLLM_MODEL_ID = "# openllm: model id"
OPENLLM_MODEL_ADAPTER_MAP = "# openllm: model adapter map"


class ModelNameFormatter(string.Formatter):
    model_keyword: t.LiteralString = "__model_name__"

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def vformat(self, format_string: str) -> str:
        return super().vformat(format_string, (), {self.model_keyword: self.model_name})

    def can_format(self, value: str) -> bool:
        try:
            self.parse(value)
            return True
        except ValueError:
            return False


class ModelIdFormatter(ModelNameFormatter):
    model_keyword: t.LiteralString = "__model_id__"


class ModelAdapterMapFormatter(ModelNameFormatter):
    model_keyword: t.LiteralString = "__model_adapter_map__"


_service_file = Path(__file__).parent.parent / "_service.py"


def write_service(llm: openllm.LLM[t.Any, t.Any], adapter_map: dict[str, str | None] | None, llm_fs: FS):
    from . import DEBUG

    model_name = llm.config["model_name"]

    logger.debug("Generating service for %s", model_name)

    with open(_service_file.__fspath__(), "r") as f:
        src_contents = f.readlines()

    # modify with model name
    for it in src_contents:
        if OPENLLM_MODEL_NAME in it:
            src_contents[src_contents.index(it)] = (
                ModelNameFormatter(model_name).vformat(it)[: -(len(OPENLLM_MODEL_NAME) + 3)] + "\n"
            )
        elif OPENLLM_MODEL_ADAPTER_MAP in it:
            src_contents[src_contents.index(it)] = (
                ModelAdapterMapFormatter(orjson.dumps(adapter_map).decode()).vformat(it)[
                    : -(len(OPENLLM_MODEL_ADAPTER_MAP) + 3)
                ]
                + "\n"
            )

    script = f"# GENERATED BY 'openllm build {model_name}'. DO NOT EDIT\n\n" + "".join(src_contents)

    if DEBUG:
        logger.info("Generated script:\n%s", script)

    llm_fs.writetext(llm.config["service_name"], script)


# NOTE: The following ins extracted from attrs internal APIs

# sentinel object for unequivocal object() getattr
_sentinel = object()


def has_own_attribute(cls: type[t.Any], attrib_name: t.Any):
    """
    Check whether *cls* defines *attrib_name* (and doesn't just inherit it).
    """
    attr = getattr(cls, attrib_name, _sentinel)
    if attr is _sentinel:
        return False

    for base_cls in cls.__mro__[1:]:
        a = getattr(base_cls, attrib_name, None)
        if attr is a:
            return False

    return True


def get_annotations(cls: type[t.Any]) -> DictStrAny:
    """
    Get annotations for *cls*.
    """
    if has_own_attribute(cls, "__annotations__"):
        return cls.__annotations__

    return DictStrAny()


_classvar_prefixes = (
    "typing.ClassVar",
    "t.ClassVar",
    "ClassVar",
    "typing_extensions.ClassVar",
)


def is_class_var(annot: str | t.Any) -> bool:
    """
    Check whether *annot* is a typing.ClassVar.

    The string comparison hack is used to avoid evaluating all string
    annotations which would put attrs-based classes at a performance
    disadvantage compared to plain old classes.
    """
    annot = str(annot)

    # Annotation can be quoted.
    if annot.startswith(("'", '"')) and annot.endswith(("'", '"')):
        annot = annot[1:-1]

    return annot.startswith(_classvar_prefixes)


def add_method_dunders(cls: type[t.Any], method_or_cls: _T, _overwrite_doc: str | None = None) -> _T:
    """
    Add __module__ and __qualname__ to a *method* if possible.
    """
    try:
        method_or_cls.__module__ = cls.__module__
    except AttributeError:
        pass

    try:
        method_or_cls.__qualname__ = ".".join((cls.__qualname__, method_or_cls.__name__))
    except AttributeError:
        pass

    try:
        method_or_cls.__doc__ = (
            _overwrite_doc or "Method or class generated by LLMConfig for class " f"{cls.__qualname__}."
        )
    except AttributeError:
        pass

    return method_or_cls


def generate_unique_filename(cls: type[t.Any], func_name: str):
    return f"<{cls.__name__} generated {func_name} {cls.__module__}." f"{getattr(cls, '__qualname__', cls.__name__)}>"


def generate_function(
    typ: type[t.Any],
    func_name: str,
    lines: list[str] | None,
    args: tuple[str, ...] | None,
    globs: dict[str, t.Any],
    annotations: dict[str, t.Any] | None = None,
):
    from . import DEBUG

    script = "def %s(%s):\n    %s\n" % (
        func_name,
        ", ".join(args) if args is not None else "",
        "\n    ".join(lines) if lines else "pass",
    )
    meth = _make_method(func_name, script, generate_unique_filename(typ, func_name), globs)
    if annotations:
        meth.__annotations__ = annotations

    if DEBUG and int(os.environ.get("OPENLLMDEVDEBUG", str(0))) > 3:
        logger.info("Generated script for %s:\n\n%s", typ, script)

    return meth


def make_env_transformer(
    cls: type[openllm.LLMConfig],
    model_name: str,
    suffix: t.LiteralString | None = None,
    default_callback: t.Callable[[str, t.Any], t.Any] | None = None,
    globs: DictStrAny | None = None,
):
    from . import dantic, field_env_key

    def identity(_: str, x_value: t.Any) -> t.Any:
        return x_value

    default_callback = identity if default_callback is None else default_callback

    globs = {} if globs is None else globs
    globs.update(
        {
            "__populate_env": dantic.env_converter,
            "__default_callback": default_callback,
            "__field_env": field_env_key,
            "__suffix": suffix or "",
            "__model_name": model_name,
        }
    )

    lines: ListStr = [
        "__env = lambda field_name: __field_env(__model_name, field_name, __suffix)",
        "return [",
        "    f.evolve(",
        "        default=__populate_env(__default_callback(f.name, f.default), __env(f.name)),",
        "        metadata={",
        "            'env': f.metadata.get('env', __env(f.name)),",
        "            'description': f.metadata.get('description', '(not provided)'),",
        "        },",
        "    )",
        "    for f in fields",
        "]",
    ]
    fields_ann = "list[attr.Attribute[t.Any]]"

    return generate_function(
        cls,
        "__auto_env",
        lines,
        args=("_", "fields"),
        globs=globs,
        annotations={"_": "type[LLMConfig]", "fields": fields_ann, "return": fields_ann},
    )
