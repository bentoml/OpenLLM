#!/usr/bin/env bash

set -ex

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

[[ -x "$(command -v docker)" ]] || (
    echo "docker not found. Make sure to have docker running to run this job."
    exit 1
)

find "${GIT_ROOT}/.github/workflows" -type f -iname '*.yml' -exec docker run -it --rm -v "${PWD}":"${PWD}" -w "${PWD}" -e RATCHET_EXP_KEEP_NEWLINES=true ghcr.io/sethvargo/ratchet:0.4.0 update {} \;
