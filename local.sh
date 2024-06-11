#!/usr/bin/env bash
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit 1

usage() {
  echo "Usage: $0 [--daemon|--tools] [--debug] [--tests]"
  echo
  echo "Options:"
  echo "  --daemon            Running as background, skip all script."
  echo "  --tools             Running related tools. Mutually exclusive with --daemon"
  echo "  --tests             Run tests environment."
  echo "  --assume-yes | -y   Assuming yes to all instructions.."
  echo "  --help | -h         Show this help message"
  echo ""
  echo "NOTE: --tools will skip setting up environment."
  echo
}

DEBUG=false
DAEMON=false
YES=false
TESTS=false
TOOLS=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --daemon)
      DAEMON=true
      shift
      ;;
    --tools)
      TOOLS=true
      YES=true
      shift
      ;;
    --tests)
      TESTS=true
      DAEMON=true
      shift
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    --assume-yes | -y)
      YES=true
      shift
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

# Ensure that --pr and --head are mutually exclusive
if [[ "${TOOLS:-}" = true && "${DAEMON:-}" = true ]]; then
  echo "Error: --tools and --daemon are mutually exclusive"
  exit 1
fi

if [ "$DAEMON" = false ] && ! $YES; then
  echo ""
  echo "The script will do the following (To skip this description, pass in -y):"
  echo "  1. It will check for uv, if doesn't exist then proceed to install it."
  echo "  2. It will then check for .python-version, if doesn't exists then symlink from .python-version-default"
  echo "  3. It will do the same for .envrc"
  echo "  4. Create a virtualenv under $GIT_ROOT/.venv based on .python-version-default"
  echo "  5. source the venv, then proceed to install all openllm libraries to this venv"
  echo "  6. By default, it will run all.sh scripts, which mainly include tools for openllm. If --daemon is passed, then this step is skipped."
  echo "  7. Optionally install tests dependencies, then run tests set."
  echo ""
  while true; do
    read -p "Do you want to proceed? (yY/nN) " yn
    case $yn in
      [Yy]*)
        break
        ;;
      [Nn]*)
        exit
        ;;
      *)
        echo "Please answer y|Y or n|N."
        ;;
    esac
  done
fi

if [ "$DEBUG" = true ]; then
  echo "running path: $0"
  set -x
fi

function run_tools {
  echo "Running tools..."
  uv tool run pre-commit install &>/dev/null
  python "$GIT_ROOT/tools/update-mypy.py"
  python "$GIT_ROOT/tools/dependencies.py"
  python "$GIT_ROOT/tools/update-config-stubs.py"
  python "$GIT_ROOT/tools/update-readme.py"
  bash "$GIT_ROOT/tools/mirror.sh"
}

if [ "$TOOLS" = true ]; then
  . "$GIT_ROOT/.venv/bin/activate"
  run_tools || exit 1
  exit 0
fi

# check if uv is installed
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="$HOME/.cargo/bin/uv"
else
  UV_BIN=$(command -v uv)
fi

# Check if .python-version file exists from GIT_ROOT, otherwise symlink from .python-version-default to .python-version
if [ ! -f "$GIT_ROOT/.python-version" ]; then
  ln -s "$GIT_ROOT/.python-version-default" "$GIT_ROOT/.python-version"
fi

if [ ! -f "$GIT_ROOT/.envrc" ]; then
  cp "$GIT_ROOT/.envrc.template" "$GIT_ROOT/.envrc"
fi

# check if there is a $GIT_ROOT/.venv directory, if not, create it
if [ ! -d "$GIT_ROOT/.venv" ]; then
  # get the python version from $GIT_ROOT/.python-version-default
  "$UV_BIN" venv -p "$(cat "$GIT_ROOT/.python-version-default")" "$GIT_ROOT/.venv"
fi

PRERELEASE=${PRERELEASE:-false}

ARGS=()
[[ "${PRERELEASE}" == "true" ]] && ARGS+=("--prerelease=allow")

"$UV_BIN" pip install "${ARGS[@]}" --editable "$GIT_ROOT/openllm-python" || true
"$UV_BIN" pip install "${ARGS[@]}" --editable "$GIT_ROOT/openllm-client"
"$UV_BIN" pip install "${ARGS[@]}" --editable "$GIT_ROOT/openllm-core"

if [ "$DAEMON" = false ]; then
  "$UV_BIN" pip install -r "$GIT_ROOT/tools/requirements.txt"
fi

. "$GIT_ROOT/.venv/bin/activate"

if [ "$DAEMON" = false ]; then
  run_tools
fi

ARGS=()
if [ "$TESTS" = true ]; then
  LOCK_FILE="/tmp/openllm_tests.lock"

  # Create the lock file
  exec {lock_fd}>"$LOCK_FILE"

  # Acquire an exclusive lock
  flock -x "$lock_fd"

  # Check if the lock file already exists
  if [ -f "$LOCK_FILE" ]; then
    echo "Another instance of the script is already running tests. Waiting for it to finish..."
    flock -s "$lock_fd"
  fi
  "$UV_BIN" pip install -r "$GIT_ROOT/tools/tests-requirements.txt"

  # TODO: uv tool run once it becomes more stable
  # [ "$DEBUG" = true ] && ARGS+=("--verbose")
  # ARGS+=$(cat "$GIT_ROOT/tools/tests-requirements.txt" | tr '\n' '\0' | sed -z 's/^/ --with /')
  python -m pytest --cov --cov-report="${COVERAGE_REPORT:-term-missing}" --cov-config="$GIT_ROOT/pyproject.toml" -vv -r aR -x "${PYTEST_ARGS:-${GIT_ROOT}/openllm-python/tests}"

  # Release the lock
  flock -u "$lock_fd"

  # Close the file descriptor
  exec {lock_fd}>&-

  # Remove the lock file
  rm -f "$LOCK_FILE"
fi
