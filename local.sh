#!/usr/bin/env bash

set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit 1

# check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Check if .python-version file exists from GIT_ROOT, otherwise symlink from .python-version-default to .python-version
if [ ! -f "$GIT_ROOT/.python-version" ]; then
  echo "Symlinking .python-version-default to .python-version"
  ln -s "$GIT_ROOT/.python-version-default" "$GIT_ROOT/.python-version"
fi

# check if there is a $GIT_ROOT/.venv directory, if not, create it
if [ ! -d "$GIT_ROOT/.venv" ]; then
  # get the python version from $GIT_ROOT/.python-version-default
  uv venv -p $(cat "$GIT_ROOT/.python-version-default") "$GIT_ROOT/.venv"
fi

. "$GIT_ROOT/.venv/bin/activate"

print_usage() {
  echo "Usage: $0"
}

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
  --help | -h)
    print_usage
    exit 0
    ;;
  *)
    print_usage
    exit 1
    ;;
  esac
  shift
done

PRERELEASE=${PRERELEASE:-false}

ARGS=()
[[ "${PRERELEASE}" == "true" ]] && ARGS+=("--prerelease=allow")

uv pip install "${ARGS[@]}" --editable "$GIT_ROOT/openllm-python"
uv pip install "${ARGS[@]}" --editable "$GIT_ROOT/openllm-client"
uv pip install "${ARGS[@]}" --editable "$GIT_ROOT/openllm-core"

echo "Instaling development dependencies..."
uv pip install -r "$GIT_ROOT/tools/requirements.txt"

pre-commit install

bash "$GIT_ROOT/all.sh"
