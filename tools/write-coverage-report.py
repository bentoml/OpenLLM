#!/usr/bin/env python3
from __future__ import annotations
from decimal import ROUND_DOWN, Decimal
from pathlib import Path
import orjson

PRECISION = Decimal(".01")

ROOT = Path(__file__).resolve().parent.parent

def main() -> int:
  coverage_summary = ROOT/"coverage-summary.json"

  coverage_data = orjson.loads(coverage_summary.read_text(encoding="utf-8"))
  total_data = coverage_data.pop("total")

  lines = ["\n", "Package | Statements\n", "------- | ----------\n",]

  for package, data in sorted(coverage_data.items()):
    statements_covered = data["statements_covered"]
    statements = data["statements"]

    rate = Decimal(statements_covered) / Decimal(statements) * 100
    rate = rate.quantize(PRECISION, rounding=ROUND_DOWN)
    lines.append(f"{package} | {100 if rate == 100 else rate}% ({statements_covered} / {statements})\n")

  total_statements_covered = total_data["statements_covered"]
  total_statements = total_data["statements"]
  total_rate = Decimal(total_statements_covered) / Decimal(total_statements) * 100
  total_rate = total_rate.quantize(PRECISION, rounding=ROUND_DOWN)
  color = "ok" if float(total_rate) >= 95 else "critical"
  lines.insert(0, f"![Code Coverage](https://img.shields.io/badge/coverage-{total_rate}%25-{color}?style=flat)\n")

  lines.append(f"**Summary** | {100 if total_rate == 100 else total_rate}% ({total_statements_covered} / {total_statements})\n")

  coverage_report = ROOT/"coverage-report.md"
  with coverage_report.open("w", encoding="utf-8") as f: f.write("".join(lines))
  return 0

if __name__ == "__main__": raise SystemExit(main())
