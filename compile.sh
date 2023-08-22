#!/usr/bin/env bash
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
HATCH_VERBOSE=3 MYPY_CONFIG_FILE_DIR="${SCRIPT_DIR}" HATCH_BUILD_HOOKS_ENABLE=1 MYPYPATH="${SCRIPT_DIR}/typings" python -m build openllm-python -w -C--global-option=--verbose "$@"
HATCH_VERBOSE=3 MYPY_CONFIG_FILE_DIR="${SCRIPT_DIR}" HATCH_BUILD_HOOKS_ENABLE=1 MYPYPATH="${SCRIPT_DIR}/typings" python -m build openllm-core -w -C--global-option=--verbose "$@"
HATCH_VERBOSE=3 MYPY_CONFIG_FILE_DIR="${SCRIPT_DIR}" HATCH_BUILD_HOOKS_ENABLE=1 MYPYPATH="${SCRIPT_DIR}/typings" python -m build openllm-client -w -C--global-option=--verbose "$@"
hatch clean
