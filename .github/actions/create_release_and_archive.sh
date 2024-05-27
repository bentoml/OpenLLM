set -o errexit -o nounset -o pipefail

TAG="${1#v}"

# Check if an argument is provided
if [ $# -eq 0 ]; then
  echo "No argument provided."
  exit 1
fi

cat >release_notes.txt <<EOF
## Installation

\`\`\`bash
pip install openllm==${TAG}
\`\`\`

To upgrade from a previous version, use the following command:
\`\`\`bash
pip install --upgrade openllm==${TAG}
\`\`\`

## Usage

To start a LLM: \`\`\`python -m openllm start HuggingFaceH4/zephyr-7b-beta\`\`\`

Find more information about this release in the [CHANGELOG.md](https://github.com/bentoml/OpenLLM/blob/main/CHANGELOG.md)

EOF
