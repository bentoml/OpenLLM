#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import importlib

_TARGET_FILE = Path(__file__).parent.parent/"src"/"openllm"/"__init__.py"
_IGNORE = ["warnings", "os", "logging", "Path"]

def main() -> int:
  with _TARGET_FILE.open("r") as f: processed = f.readlines()
  var = [f'"{v}"' for v in filter(lambda v: v not in _IGNORE and not v.startswith('_'), dir(importlib.import_module('openllm')))]
  # Add some of the static external variable here as well
  var += [f'"COMPILED"']
  processed = processed[:-1] + [f"__all__=[{','.join(var)}]\n"]
  with _TARGET_FILE.open("w") as f: f.writelines(processed)
  return 0

if __name__ == "__main__": raise SystemExit(main())
