from __future__ import annotations
import os, typing as t
from hatchling.metadata.plugin.interface import MetadataHookInterface

class CustomMetadataHook(MetadataHookInterface):
  def update(self, metadata: dict[str, t.Any]) -> None:
    if os.environ.get("HATCH_ENV_ACTIVE", "not-dev") != "dev": metadata["dependencies"] = [f"openllm[opt,chatglm,fine-tune]=={metadata['version']}"]
