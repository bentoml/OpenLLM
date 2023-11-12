"""CLI entrypoint for OpenLLM.

Usage:
    openllm --help

To start any OpenLLM model:
    openllm start <model_name> --options ...
"""

if __name__ == '__main__':
  from openllm_cli.entrypoint import cli

  cli()
