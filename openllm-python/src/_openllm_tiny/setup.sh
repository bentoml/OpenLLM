#!/usr/bin/env bash
set -Eexuo pipefail

BASEDIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]:-$0}")" &>/dev/null && pwd 2>/dev/null)"
PARENT_DIR="$(dirname -- "$BASEDIR")"
WHEELS_DIR="${PARENT_DIR}/python/wheels"
pushd "${PARENT_DIR}/python" &>/dev/null
shopt -s nullglob
targzs=($WHEELS_DIR/*.tar.gz)
if [ ${#targzs[@]} -gt 0 ]; then
  echo "Installing tar.gz packaged in Bento.."
  pip3 install --no-color --progress-bar off "${targzs[@]}"
fi
popd &>/dev/null
