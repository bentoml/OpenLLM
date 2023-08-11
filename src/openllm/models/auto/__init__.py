from __future__ import annotations
import sys
import typing as t
import openllm
from openllm.utils import LazyModule
from openllm.utils import is_flax_available
from openllm.utils import is_tf_available
from openllm.utils import is_torch_available
from openllm.utils import is_vllm_available

_import_structure: dict[str, list[str]] = {"configuration_auto": ["AutoConfig", "CONFIG_MAPPING", "CONFIG_MAPPING_NAMES"], "modeling_auto": ["MODEL_MAPPING_NAMES"], "modeling_flax_auto": ["MODEL_FLAX_MAPPING_NAMES"], "modeling_tf_auto": ["MODEL_TF_MAPPING_NAMES"], "modeling_vllm_auto": ["MODEL_VLLM_MAPPING_NAMES"]}
try:
  if not is_torch_available(): raise openllm.exceptions.MissingDependencyError
except openllm.exceptions.MissingDependencyError: pass
else: _import_structure["modeling_auto"].extend(["AutoLLM", "MODEL_MAPPING"])
try:
  if not is_vllm_available(): raise openllm.exceptions.MissingDependencyError
except openllm.exceptions.MissingDependencyError: pass
else: _import_structure["modeling_vllm_auto"].extend(["AutoVLLM", "MODEL_VLLM_MAPPING"])
try:
  if not is_flax_available(): raise openllm.exceptions.MissingDependencyError
except openllm.exceptions.MissingDependencyError: pass
else: _import_structure["modeling_flax_auto"].extend(["AutoFlaxLLM", "MODEL_FLAX_MAPPING"])
try:
  if not is_tf_available(): raise openllm.exceptions.MissingDependencyError
except openllm.exceptions.MissingDependencyError: pass
else: _import_structure["modeling_tf_auto"].extend(["AutoTFLLM", "MODEL_TF_MAPPING"])
if t.TYPE_CHECKING:
  from .configuration_auto import CONFIG_MAPPING as CONFIG_MAPPING
  from .configuration_auto import CONFIG_MAPPING_NAMES as CONFIG_MAPPING_NAMES
  from .configuration_auto import AutoConfig as AutoConfig
  from .modeling_auto import MODEL_MAPPING_NAMES as MODEL_MAPPING_NAMES
  from .modeling_flax_auto import MODEL_FLAX_MAPPING_NAMES as MODEL_FLAX_MAPPING_NAMES
  from .modeling_tf_auto import MODEL_TF_MAPPING_NAMES as MODEL_TF_MAPPING_NAMES
  from .modeling_vllm_auto import MODEL_VLLM_MAPPING_NAMES as MODEL_VLLM_MAPPING_NAMES
  try:
    if not is_torch_available(): raise openllm.exceptions.MissingDependencyError
  except openllm.exceptions.MissingDependencyError: pass
  else: from .modeling_auto import MODEL_MAPPING as MODEL_MAPPING, AutoLLM as AutoLLM
  try:
    if not is_vllm_available(): raise openllm.exceptions.MissingDependencyError
  except openllm.exceptions.MissingDependencyError: pass
  else: from .modeling_vllm_auto import MODEL_VLLM_MAPPING as MODEL_VLLM_MAPPING, AutoVLLM as AutoVLLM
  try:
    if not is_flax_available(): raise openllm.exceptions.MissingDependencyError
  except openllm.exceptions.MissingDependencyError: pass
  else: from .modeling_flax_auto import MODEL_FLAX_MAPPING as MODEL_FLAX_MAPPING, AutoFlaxLLM as AutoFlaxLLM
  try:
    if not is_tf_available(): raise openllm.exceptions.MissingDependencyError
  except openllm.exceptions.MissingDependencyError: pass
  else: from .modeling_tf_auto import MODEL_TF_MAPPING as MODEL_TF_MAPPING, AutoTFLLM as AutoTFLLM
else: sys.modules[__name__] = LazyModule(__name__, globals()["__file__"], _import_structure, module_spec=__spec__)
