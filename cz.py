#!/usr/bin/env python3
import itertools, os, token, tokenize

TOKEN_WHITELIST = [token.OP, token.NAME, token.NUMBER, token.STRING]

_ignored = ['_version.py']

_dir_package = {'openllm-python': 'openllm', 'openllm-core': 'openllm_core', 'openllm-client': 'openllm_client'}


def run_cz(args):
  from tabulate import tabulate

  headers = ['Name', 'Lines', 'Tokens/Line']
  table = []
  package = _dir_package[args.dir]
  for path, _, files in os.walk(os.path.join(args.dir, 'src', package)):
    for name in files:
      if not name.endswith('.py') or name in _ignored:
        continue
      filepath = os.path.join(path, name)
      with tokenize.open(filepath) as file_:
        tokens = [t for t in tokenize.generate_tokens(file_.readline) if t.type in TOKEN_WHITELIST]
        token_count, line_count = len(tokens), len(set([t.start[0] for t in tokens]))
        table.append([
          filepath.replace(os.path.join(args.dir, 'src'), ''),
          line_count,
          token_count / line_count if line_count != 0 else 0,
        ])
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
  return 0


if __name__ == '__main__':
  import argparse, importlib.util

  if importlib.util.find_spec('tabulate') is None:
    raise SystemExit('tabulate not installed. Install with `pip install tabulate`')

  parser = argparse.ArgumentParser()
  parser.add_argument(
    '--dir',
    choices=['openllm-python', 'openllm-core', 'openllm-client'],
    help='directory to check',
    default='openllm-python',
    required=False,
  )
  raise SystemExit(run_cz(parser.parse_args()))
