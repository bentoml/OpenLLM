#!/usr/bin/env bash

set -eo pipefail

GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "$GIT_ROOT" || exit 1

mirror() {
  cp $1 $2
}

mirror README.md openllm-python/README.md
mirror LICENSE.md openllm-python/LICENSE.md
mirror CHANGELOG.md openllm-python/CHANGELOG.md
