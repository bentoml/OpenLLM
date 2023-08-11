from __future__ import annotations

import pytest

import openllm

@pytest.mark.parametrize("name", ["AutoConfig", "AutoLLM", "AutoVLLM", "AutoFlaxLLM", "AutoTFLLM", "LLM", "LLMRunner", "LLMRunnable", "LLMEmbeddings",
                                  "Runner", "client", "exceptions", "bundle", "build", "cli", "ggml", "transformers", "import_model", "infer_auto_class",
                                  "infer_quantisation_config", "models", "list_models", "start", "start_grpc", "build", "serialisation"])
def test_compiled_imports(name: str):
  assert getattr(openllm, name) is not None
