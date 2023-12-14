from __future__ import annotations
import openllm, transformers, typing as t


def load_model(llm: openllm.LLM, config: transformers.PretrainedConfig, **attrs: t.Any): ...
