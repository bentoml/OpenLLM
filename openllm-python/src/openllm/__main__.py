'''CLI entrypoint for OpenLLM.

Usage:
    openllm --help

To start any OpenLLM model:
    openllm start <model_name> --options ...
'''
from __future__ import annotations

if __name__ == '__main__':
  from openllm.cli.entrypoint import cli
  cli()
