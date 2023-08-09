#!/usr/bin/env bash
GIT_ROOT="$(git rev-parse --show-toplevel)"
cd "$GIT_ROOT" || exit 1
rm -r ./**/*.so
