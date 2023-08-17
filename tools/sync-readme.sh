#!/usr/bin/env bash

if ! git diff --quiet README.md; then
    cp README.md openllm-python/README.md
    exit 1
else
    echo "README.md is up to date"
    exit 0
fi
