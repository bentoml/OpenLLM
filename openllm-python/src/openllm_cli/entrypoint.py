# mainly for backward compatible entrypoint.
from _openllm_tiny._entrypoint import cli as cli

if __name__ == '__main__':
  cli()
