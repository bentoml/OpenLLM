#!/usr/bin/env bash

set -e

DEBUG=${DEBUG:-false}
[[ "${DEBUG}" == "true" ]] && set -x

GIT_ROOT=$(git rev-parse --show-toplevel)

cd "$GIT_ROOT" || exit 1

[[ -x "$(command -v docker)" ]] || (
    echo "docker not found. Make sure to have docker running to run this job."
    exit 1
)

docker version &>/dev/null || (
  echo "docker is not healthy. Make sure to have docker running"
  exit 1
)

[[ -z "${ACTIONS_TOKEN}" ]] && (
    echo "ACTIONS_TOKEN not found. Make sure to have ACTIONS_TOKEN set to run this job."
    exit 1
)

find "${GIT_ROOT}/.github/workflows" -type f -iname '*.yml' -exec docker run --rm -v "${PWD}":"${PWD}" -w "${PWD}" -e ACTIONS_TOKEN -e RATCHET_EXP_KEEP_NEWLINES=true ghcr.io/sethvargo/ratchet:0.4.0 update {} \;
