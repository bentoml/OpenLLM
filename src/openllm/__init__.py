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
"""
OpenLLM
=======

OpenLLM: Your one stop-and-go-solution for serving any Open Large-Language Model

- StableLM, Llama, Alpaca, Dolly, Flan-T5, and more
- Powered by BentoML 🍱 + HuggingFace 🤗
"""
from __future__ import annotations

import typing as t

from . import utils as utils
from .__about__ import __version__ as __version__
from .exceptions import MissingDependencyError

_import_structure = {
    "_llm": ["LLM", "Runner"],
    "cli": ["start", "start_grpc"],
    "_configuration": ["LLMConfig"],
    "_package": ["build"],
    "exceptions": [],
    "_schema": ["PromptTemplate", "GenerationInput", "GenerationOutput"],
    "utils": [],
    "models": [],
    "client": [],
    # NOTE: models
    "models.auto": ["AutoConfig", "CONFIG_MAPPING"],
    "models.flan_t5": ["FlanT5Config"],
    "models.dolly_v2": ["DollyV2Config"],
    "models.chatglm": ["ChatGLMConfig"],
}

try:
    if not utils.is_torch_available():
        raise MissingDependencyError
except MissingDependencyError:
    from .utils import dummy_pt_objects

    _import_structure["utils.dummy_pt_objects"] = [name for name in dir(dummy_pt_objects) if not name.startswith("_")]
else:
    _import_structure["models.flan_t5"].extend(["FlanT5"])
    _import_structure["models.dolly_v2"].extend(["DollyV2"])
    _import_structure["models.chatglm"].extend(["ChatGLM"])
    _import_structure["models.auto"].extend(["AutoLLM", "MODEL_MAPPING_NAMES", "MODEL_MAPPING"])

try:
    if not utils.is_flax_available():
        raise MissingDependencyError
except MissingDependencyError:
    from .utils import dummy_flax_objects

    _import_structure["utils.dummy_flax_objects"] = [
        name for name in dir(dummy_flax_objects) if not name.startswith("_")
    ]
else:
    _import_structure["models.flan_t5"].extend(["FlaxFlanT5"])
    _import_structure["models.auto"].extend(["AutoFlaxLLM", "MODEL_FLAX_MAPPING_NAMES", "MODEL_FLAX_MAPPING"])

try:
    if not utils.is_tf_available():
        raise MissingDependencyError
except MissingDependencyError:
    from .utils import dummy_tf_objects

    _import_structure["utils.dummy_tf_objects"] = [name for name in dir(dummy_tf_objects) if not name.startswith("_")]
else:
    _import_structure["models.flan_t5"].extend(["TFFlanT5"])
    _import_structure["models.auto"].extend(["AutoTFLLM", "MODEL_TF_MAPPING_NAMES", "MODEL_TF_MAPPING"])


# declaration for OpenLLM-related modules
if t.TYPE_CHECKING:
    from . import cli as cli
    from . import client as client
    from . import exceptions as exceptions
    from . import models as models
    # Specific types import
    from ._configuration import LLMConfig as LLMConfig
    from ._llm import LLM as LLM
    from ._llm import Runner as Runner
    from ._package import build as build
    from ._schema import GenerationInput as GenerationInput
    from ._schema import GenerationOutput as GenerationOutput
    from ._schema import PromptTemplate as PromptTemplate
    from .cli import start as start
    from .cli import start_grpc as start_grpc
    from .models.auto import CONFIG_MAPPING as CONFIG_MAPPING
    from .models.auto import AutoConfig as AutoConfig
    from .models.chatglm import ChatGLMConfig as ChatGLMConfig
    from .models.dolly_v2 import DollyV2Config as DollyV2Config
    from .models.flan_t5 import FlanT5Config as FlanT5Config

    try:
        if not utils.is_torch_available():
            raise MissingDependencyError
    except MissingDependencyError:
        from .utils.dummy_pt_objects import *
    else:
        from .models.auto import MODEL_MAPPING as MODEL_MAPPING
        from .models.auto import MODEL_MAPPING_NAMES as MODEL_MAPPING_NAMES
        from .models.auto import AutoLLM as AutoLLM
        from .models.chatglm import ChatGLM as ChatGLM
        from .models.dolly_v2 import DollyV2 as DollyV2
        from .models.flan_t5 import FlanT5 as FlanT5

    try:
        if not utils.is_flax_available():
            raise MissingDependencyError
    except MissingDependencyError:
        from .utils.dummy_flax_objects import *
    else:
        from .models.auto import MODEL_FLAX_MAPPING as MODEL_FLAX_MAPPING
        from .models.auto import \
            MODEL_FLAX_MAPPING_NAMES as MODEL_FLAX_MAPPING_NAMES
        from .models.auto import AutoFlaxLLM as AutoFlaxLLM
        from .models.flan_t5 import FlaxFlanT5 as FlaxFlanT5

    try:
        if not utils.is_tf_available():
            raise MissingDependencyError
    except MissingDependencyError:
        from .utils.dummy_tf_objects import *
    else:
        from .models.auto import MODEL_TF_MAPPING as MODEL_TF_MAPPING
        from .models.auto import \
            MODEL_TF_MAPPING_NAMES as MODEL_TF_MAPPING_NAMES
        from .models.auto import AutoTFLLM as AutoTFLLM
        from .models.flan_t5 import TFFlanT5 as TFFlanT5

else:
    import sys

    sys.modules[__name__] = utils.LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
        extra_objects={"__version__": __version__},
    )
