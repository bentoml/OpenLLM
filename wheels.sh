#!/usr/bin/env bash

set -eo pipefail

GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "$GIT_ROOT" || exit 1

mkdir -p dist

pushd openllm-client &>/dev/null

python -m build -w && mv dist/* ../dist

popd &>/dev/null

pushd openllm-core &>/dev/null

python -m build -w && mv dist/* ../dist

popd &>/dev/null

pushd openllm-python &>/dev/null

python -m build -w && mv dist/* ../dist

popd &>/dev/null
