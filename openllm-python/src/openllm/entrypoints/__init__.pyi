"""Entrypoint for all third-party apps.

Currently support OpenAI, Cohere compatible API.

Each module should implement the following API:

- `mount_to_svc(svc: bentoml.Service, llm: openllm.LLM[M, T]) -> bentoml.Service: ...`
"""

from bentoml import Service
from openllm_core._typing_compat import M, T

from . import cohere as cohere, hf as hf, openai as openai
from .._llm import LLM

def mount_entrypoints(svc: Service, llm: LLM[M, T]) -> Service: ...
