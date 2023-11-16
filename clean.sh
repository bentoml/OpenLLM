#!/usr/bin/env bash
GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "$GIT_ROOT" || exit 1
find . -type f -iname "*.so" -exec rm -f {} \;
find . -type d -iname "__pycache__" -exec rm -rf {} \; -prune
find . -type d -iname "node_modules" -exec rm -rf "{}" \; -prune
