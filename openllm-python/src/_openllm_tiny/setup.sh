#!/usr/bin/env bash
set -Eexuo pipefail

FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip3 install --no-build-isolation --no-color --progress-bar off flash-attn==2.5.7 || true
