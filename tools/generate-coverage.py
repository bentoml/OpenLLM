#!/usr/bin/env python3
from __future__ import annotations
from collections import defaultdict
from pathlib import Path

import orjson
from lxml import etree
ROOT = Path(__file__).resolve().parent.parent

PACKAGES = {'openllm-python/src/openllm/': 'openllm'}

def main() -> int:
  coverage_report = ROOT / 'coverage.xml'
  root = etree.fromstring(coverage_report.read_text())

  raw_package_data: defaultdict[str, dict[str, int]] = defaultdict(lambda: {'hits': 0, 'misses': 0})
  for package in root.find('packages'):
    for module in package.find('classes'):
      filename = module.attrib['filename']
      for relative_path, package_name in PACKAGES.items():
        if filename.startswith(relative_path):
          data = raw_package_data[package_name]
          break
      else:
        message = f'unknown package: {module}'
        raise ValueError(message)

      for line in module.find('lines'):
        if line.attrib['hits'] == '1': data['hits'] += 1
        else: data['misses'] += 1

  total_statements_covered = 0
  total_statements = 0
  coverage_data = {}
  for package_name, data in sorted(raw_package_data.items()):
    statements_covered = data['hits']
    statements = statements_covered + data['misses']
    total_statements_covered += statements_covered
    total_statements += statements
    coverage_data[package_name] = {'statements_covered': statements_covered, 'statements': statements}
  coverage_data['total'] = {'statements_covered': total_statements_covered, 'statements': total_statements}

  coverage_summary = ROOT / 'coverage-summary.json'
  coverage_summary.write_text(orjson.dumps(coverage_data, option=orjson.OPT_INDENT_2).decode(), encoding='utf-8')
  return 0

if __name__ == '__main__': raise SystemExit(main())
