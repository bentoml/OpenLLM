from typing import Dict, Optional, Tuple

from fs.base import FS
from typing_extensions import LiteralString

from bentoml import Bento, Tag
from bentoml._internal.bento import BentoStore
from bentoml._internal.bento.build_config import DockerOptions, PythonOptions
from bentoml._internal.models.model import ModelStore
from openllm_core._typing_compat import LiteralQuantise, LiteralSerialisation, M, T

from .._llm import LLM

def build_editable(path: str, package: LiteralString) -> Optional[str]: ...
def construct_python_options(
  llm: LLM[M, T],
  llm_fs: FS,
  extra_dependencies: Optional[Tuple[str, ...]] = ...,
  adapter_map: Optional[Dict[str, str]] = ...,
) -> PythonOptions: ...
def construct_docker_options(
  llm: LLM[M, T],
  llm_fs: FS,
  quantize: Optional[LiteralQuantise],
  adapter_map: Optional[Dict[str, str]],
  dockerfile_template: Optional[str],
  serialisation: LiteralSerialisation,
) -> DockerOptions: ...
def create_bento(
  bento_tag: Tag,
  llm_fs: FS,
  llm: LLM[M, T],
  quantize: Optional[LiteralQuantise],
  dockerfile_template: Optional[str],
  adapter_map: Optional[Dict[str, str]] = ...,
  extra_dependencies: Optional[Tuple[str, ...]] = ...,
  serialisation: Optional[LiteralSerialisation] = ...,
  _bento_store: BentoStore = ...,
  _model_store: ModelStore = ...,
) -> Bento: ...
