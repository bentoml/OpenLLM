#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import itertools
import importlib

_client_all = Path(__file__).parent.parent/"src"/"openllm"/"client.py"

def main() -> int:
  mod = importlib.import_module("openllm.client")
  _all = [f'"{i}"' for i in itertools.chain.from_iterable(getattr(mod, "_import_structure").values())]
  with _client_all.open("r") as f: processed = f.readlines()
  processed = processed[:-1] + [f"__all__=[{','.join(sorted(_all))}]\n"]
  with _client_all.open("w") as f: f.writelines(processed)
  return 0

if __name__ == "__main__": raise SystemExit(main())
