#!/usr/bin/env python3
from __future__ import annotations
import os
from pathlib import Path

_TARGET_FILE = Path(__file__).parent.parent / 'openllm-python' / 'src' / 'openllm' / 'models' / '__init__.py'

def create_module_import() -> str:
  r = [f'"{p.name}"' for p in _TARGET_FILE.parent.glob('*/') if p.name not in ['__pycache__', '__init__.py', '.DS_Store']]
  return f"_MODELS:set[str]={{{', '.join(sorted(r))}}}"

def create_stubs_import() -> list[str]:
  return [
      'if t.TYPE_CHECKING:from . import ' +
      ','.join([f'{p.name} as {p.name}' for p in sorted(_TARGET_FILE.parent.glob('*/')) if p.name not in {'__pycache__', '__init__.py', '.DS_Store'}]),
      '__lazy=LazyModule(__name__, globals()["__file__"], {k: [] for k in _MODELS})', '__all__=__lazy.__all__', '__dir__=__lazy.__dir__',
      '__getattr__=__lazy.__getattr__\n'
  ]

def main() -> int:
  _path = os.path.join(os.path.basename(os.path.dirname(__file__)), os.path.basename(__file__))
  with _TARGET_FILE.open('w') as f:
    f.writelines('\n'.join([
        f'# This file is generated by {_path}. DO NOT EDIT MANUALLY!', f'# To update this, run ./{_path}', 'from __future__ import annotations',
        'import typing as t', 'from openllm_core.utils import LazyModule',
        create_module_import(), *create_stubs_import(),
    ]))
  return 0

if __name__ == '__main__': raise SystemExit(main())
