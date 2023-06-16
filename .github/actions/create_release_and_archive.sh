#!/usr/bin/env bash
# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

Find more information about this release in the [CHANGELOG.md](https://github.com/bentoml/OpenLLM/blob/main/CHANGELOG.md)

EOF
