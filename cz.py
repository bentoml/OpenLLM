#!/usr/bin/env python3
from __future__ import annotations
import itertools
import os
import token
import tokenize

from tabulate import tabulate


TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]


def run_cz(dir: str, package: str):
  headers = ['Name', 'Lines', 'Tokens/Line']
  table = []
  for path, _, files in os.walk(os.path.join(dir, 'src', package)):
    for name in files:
      if not name.endswith('.py'):
        continue
      filepath = os.path.join(path, name)
      with tokenize.open(filepath) as file_:
        tokens = [t for t in tokenize.generate_tokens(file_.readline) if t.type in TOKEN_WHITELIST]
        token_count, line_count = len(tokens), len(set([t.start[0] for t in tokens]))
        table.append(
          [
            filepath.replace(os.path.join(dir, 'src'), ''),
            line_count,
            token_count / line_count if line_count != 0 else 0,
          ]
        )
  print(f'\n{"=" * 80}\n')
  print(tabulate([headers, *sorted(table, key=lambda x: -x[1])], headers='firstrow', floatfmt='.1f') + '\n')
  print(
    tabulate(
      [
        (dir_name, sum([x[1] for x in group]))
        for dir_name, group in itertools.groupby(
          sorted([(x[0].rsplit('/', 1)[0], x[1]) for x in table]), key=lambda x: x[0]
        )
      ],
      headers=['Directory', 'LOC'],
      floatfmt='.1f',
    )
  )
  print(f'total line count for {package}: {sum([x[1] for x in table])}\n')


def main() -> int:
  run_cz('openllm-python', 'openllm')
  run_cz('openllm-core', 'openllm_core')
  run_cz('openllm-client', 'openllm_client')
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
