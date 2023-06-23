from __future__ import annotations

import typing as t

import openllm


_import_structure = {
    "configuration_bart": ["BartConfig", "START_BART_COMMAND_DOCSTRING", "DEFAULT_PROMPT_TEMPLATE"],
}

try:
    if not openllm.utils.is_torch_available():
        raise openllm.exceptions.MissingDependencyError
except openllm.exceptions.MissingDependencyError:
    pass
else:
    _import_structure["modeling_bart"] = ["Bart"]
    
try:
    if not openllm.utils.is_tf_available():
        raise openllm.exceptions.MissingDependencyError
except openllm.exceptions.MissingDependencyError:
    pass
else:
    _import_structure["modeling_tf_bart"] = ["TFBart"]

if t.TYPE_CHECKING:
    from .configuration_bart import DEFAULT_PROMPT_TEMPLATE as DEFAULT_PROMPT_TEMPLATE
    from .configuration_bart import START_BART_COMMAND_DOCSTRING as START_BART_COMMAND_DOCSTRING
    from .configuration_bart import BartConfig as BartConfig

    try:
        if not openllm.utils.is_torch_available():
            raise openllm.exceptions.MissingDependencyError
    except openllm.exceptions.MissingDependencyError:
        pass
    else:
        from .modeling_bart import Bart as Bart

    try:
        if not openllm.utils.is_tf_available():
            raise openllm.exceptions.MissingDependencyError
    except openllm.exceptions.MissingDependencyError:
        pass
    else:
        from .modeling_tf_bart import TFBart as TFBart
else:
    import sys

    sys.modules[__name__] = openllm.utils.LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )