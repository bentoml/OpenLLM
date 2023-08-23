#!/usr/bin/env bash

set -ex -o pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

cd "$SCRIPT_DIR" || exit 1

if ! command -v node @ >&1 > /dev/null; then
  echo "Cannot find 'node' executable in PATH. Make sure to have Node.js setup. Refer to"
fi

if ! command -v pnpm @ >&1 > /dev/null; then
    curl -fsSL https://get.pnpm.io/install.sh | sh -
fi

if ! command -v clojure @ >&1 > /dev/null; then
    curl -fsSL https://github.com/clojure/brew-install/releases/latest/download/posix-install.sh | bash -
fi

if ! command -v hatch @ >&1 > /dev/null; then
    echo "ERROR: hatch not installed. Aborting..."
    exit 1
fi

pnpm i && pnpm run dev
