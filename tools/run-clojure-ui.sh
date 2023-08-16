#!/usr/bin/env bash

set -ex -o pipefail

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

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

pushd contrib/clojure > /dev/null
pnpm install
popd > /dev/null

pnpm run -C "$GIT_ROOT"/contrib/clojure dev
