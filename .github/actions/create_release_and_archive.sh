#!/usr/bin/env bash

set -o errexit -o nounset -o pipefail

# Set by GH actions, see
# https://docs.github.com/en/actions/learn-github-actions/environment-variables#default-environment-variables
TAG=${GITHUB_REF_NAME#v}
PREFIX="openllm-${TAG}"
ARCHIVE="openllm-${TAG}.tar.gz"

git archive --format=tar --prefix="${PREFIX}/" "v${TAG}" | gzip > "${ARCHIVE}"
cat > release_notes.txt << EOF
## Installation

\`\`\`bash
pip install openllm==${TAG}
\`\`\`

To upgrade from a previous version, use the following command:
\`\`\`bash
pip install --upgrade openllm==${TAG}
\`\`\`

## Usage

All available models: \`\`\`python -m openllm.models\`\`\`

To start a LLM: \`\`\`python -m openllm start dolly-v2\`\`\`

EOF
