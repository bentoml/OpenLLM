set -o errexit -o nounset -o pipefail

TAG="${1#v}"

# Check if an argument is provided
if [ $# -eq 0 ]; then
    echo "No argument provided."
    exit 1
fi

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

All available models: \`\`\`openllm models\`\`\`

To start a LLM: \`\`\`python -m openllm start opt\`\`\`

To run OpenLLM within a container environment (requires GPUs): \`\`\`docker run --gpus all -it -P ghcr.io/bentoml/openllm:${TAG} start opt\`\`\`

To run OpenLLM Clojure UI (community-maintained): \`\`\`docker run -p 8420:80 ghcr.io/bentoml/openllm-ui-clojure:${TAG}\`\`\`

Find more information about this release in the [CHANGELOG.md](https://github.com/bentoml/OpenLLM/blob/main/CHANGELOG.md)

EOF
