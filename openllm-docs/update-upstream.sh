#!/usr/bin/env bash

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
GIT_ROOT=$(git rev-parse --show-toplevel)
cd "$SCRIPT_DIR" || exit 1

if [ "${#}" -gt 0 ]; then
  echo "Usage: $0"
  exit 1
fi

if ! command -v node @ >&1 > /dev/null; then
  echo "Cannot find 'node' executable in PATH. Make sure to have Node.js setup"
fi

if ! command -v pnpm @ >&1 > /dev/null; then
    curl -fsSL https://get.pnpm.io/install.sh | sh -
fi

BRANCH="v4"

if [[ ! -d "$SCRIPT_DIR/quartz.git" ]]; then
  git clone --bare --branch="$BRANCH" https://github.com/jackyzha0/quartz.git
else
  echo "Updating quartz.git..."
  git --git-dir=quartz.git fetch origin +refs/heads/*:refs/heads/* --prune
fi

echo "Setup sparse checkout for quartz."
git --git-dir=quartz.git --work-tree=. sparse-checkout init --no-cone
cat > quartz.git/info/sparse-checkout <<EOF
/*
!/.gitignore
!/LICENSE.txt
!/CODE_OF_CONDUCT.md
!/.gitattributes
!/content
!/.github
!/package-lock.json
EOF
git --git-dir=quartz.git --work-tree=. sparse-checkout reapply
echo "checkout quartz@$BRANCH:$(git rev-parse HEAD)"
if [[ ! -f "$SCRIPT_DIR/.docs-checkout" ]]; then
  touch .docs-checkout
  git --git-dir=quartz.git --work-tree=. checkout HEAD -f
else
  git --git-dir=quartz.git --work-tree=. checkout HEAD -- quartz .npmrc .prettierignore .prettierrc package.json tsconfig.json globals.d.ts index.d.ts
fi

pushd "$GIT_ROOT" > /dev/null
find "$SCRIPT_DIR/patches" -type f -name "*.patch" -print0 | sort -z | xargs -0 -n1 git --git-dir=quartz.git --work-tree=. apply --whitespace=nowarn --ignore-space-change --ignore-whitespace

git update-index -q --refresh
if ! git diff-index --quiet HEAD --; then
  echo "The following files are modified in the working tree:"
  git diff-index --name-only HEAD --
  echo "Please manually commit these changes and adjust accordingly."
fi
popd > /dev/null
