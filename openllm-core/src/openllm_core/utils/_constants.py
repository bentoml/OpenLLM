import sys, os

# See https://github.com/bentoml/BentoML/blob/a59750c5044bab60b6b3765e6c17041fd8984712/src/bentoml_cli/env.py#L17
DEBUG_ENV_VAR = 'BENTOML_DEBUG'
QUIET_ENV_VAR = 'BENTOML_QUIET'
# https://github.com/grpc/grpc/blob/master/doc/environment_variables.md
GRPC_DEBUG_ENV_VAR = 'GRPC_VERBOSITY'
WARNING_ENV_VAR = 'OPENLLM_DISABLE_WARNING'
DEV_DEBUG_VAR = 'DEBUG'

ENV_VARS_TRUE_VALUES = {'1', 'ON', 'YES', 'TRUE'}
OPENLLM_DEV_BUILD = 'OPENLLM_DEV_BUILD'


def check_bool_env(env: str, default: bool = True):
  v = os.getenv(env, default=str(default)).upper()
  if v.isdigit():
    return bool(int(v))  # special check for digits
  return v in ENV_VARS_TRUE_VALUES


# Special debug flag controled via DEBUG
DEBUG = sys.flags.dev_mode or (not sys.flags.ignore_environment and check_bool_env(DEV_DEBUG_VAR, default=False))
# Whether to show the codenge for debug purposes
SHOW_CODEGEN = (
  DEBUG and os.environ.get(DEV_DEBUG_VAR, str(0)).isdigit() and int(os.environ.get(DEV_DEBUG_VAR, str(0))) > 3
)
# MYPY is like t.TYPE_CHECKING, but reserved for Mypy plugins
MYPY = False
