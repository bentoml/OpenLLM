#!/usr/bin/env bash

set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit 1

# check if uv is installed
if ! command -v uv > /dev/null 2>&1; then
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
  echo "Usage: $0 [OPTIONS]"
  echo "Options:"
  echo "  -e, -E, --ext  Specify extensions for OpenLLM. Can be used multiple times or as a comma-separated list."
  echo "                  Example: $0 -e ext1,ext2"
  echo "                  Example: $0 --ext ext1 --ext ext2"
  echo ""
  echo "This script installs various components with optional extensions."
}

split_csv() {
  local IFS=','
  read -ra ADDR <<< "$1"
  for i in "${ADDR[@]}"; do
    EXTENSIONS+=("$i")
  done
}

# Function to validate extensions
validate_extensions() {
  uv pip install tomlkit pre-commit mypy
  local valid_extensions
  valid_extensions=$(python -c "
import tomlkit

with open('$GIT_ROOT/openllm-python/pyproject.toml', 'r') as file:
    data = tomlkit.load(file)
    optional_dependencies = data['project']['optional-dependencies']
    print(' '.join(optional_dependencies.keys()))
  ")

  COMMENT="[${valid_extensions[*]}]"
  COMMENT=${COMMENT// /,} # Replace spaces with commas
  for ext in "${EXTENSIONS[@]}"; do
    if ! [[ $valid_extensions =~ (^|[[:space:]])$ext($|[[:space:]]) ]]; then
      echo "Invalid extension: $ext. Available extensions are: $COMMENT"
      exit 1
    fi
  done
}

EXTENSIONS=()

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --extensions|-e|-E|--ext)
      if [[ -n $2 && $2 != -* ]]; then
        split_csv "$2"
        shift
      else
        print_usage
        exit 1
      fi
      ;;
    --help|-h)
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

validate_extensions

# Check if the EXTENSIONS array is empty
if [ ${#EXTENSIONS[@]} -eq 0 ]; then
  echo "No extensions specified"
  EXTENSIONS_STR=""
else
  echo "Installing extensions: ${EXTENSIONS[*]}"
  EXTENSIONS_STR="[${EXTENSIONS[*]}]"
  EXTENSIONS_STR=${EXTENSIONS_STR// /,} # Replace spaces with commas
fi

uv pip install --editable "$GIT_ROOT/openllm-python$EXTENSIONS_STR"
uv pip install --editable "$GIT_ROOT/openllm-client"
uv pip install --editable "$GIT_ROOT/openllm-core"

echo "Instaling development dependencies..."
uv pip install -r "$GIT_ROOT/tools/requirements.txt"

pre-commit install

bash "$GIT_ROOT/all.sh"
