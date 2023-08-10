#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import itertools
import importlib

_main_init = Path(__file__).parent.parent/"src"/"openllm"/"__init__.py"
_client_all = Path(__file__).parent.parent/"src"/"openllm"/"client.py"
_IGNORE = ["warnings", "os", "logging", "Path"]

def main() -> int:
  with _main_init.open("r") as f: processed = f.readlines()
  var = [f'"{v}"' for v in filter(lambda v: v not in _IGNORE and not v.startswith('_'), dir(importlib.import_module('openllm')))]
  # Add some of the static external variable here as well
  var += [f'"COMPILED"']
  processed = processed[:-1] + [f"__all__=[{','.join(var)}]\n"]
  with _main_init.open("w") as f: f.writelines(processed)

  mod = importlib.import_module("openllm.client")
  _all = [f'"{i}"' for i in itertools.chain.from_iterable(getattr(mod, "_import_structure").values())]
  with _client_all.open("r") as f: processed = f.readlines()
  processed = processed[:-1] + [f"__all__=[{','.join(_all)}]\n"]
  with _client_all.open("w") as f: f.writelines(processed)
  return 0

if __name__ == "__main__": raise SystemExit(main())
