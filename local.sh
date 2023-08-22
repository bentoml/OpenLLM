#!/usr/bin/env bash

set -ex

GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$GIT_ROOT" || exit 1

pip install -e "$GIT_ROOT/openllm-core" -v
pip install -e "$GIT_ROOT/openllm-client" -v
pip install -e "$GIT_ROOT/openllm-python" -v
