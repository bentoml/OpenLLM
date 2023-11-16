#!/usr/bin/env python3
import os
import shutil
import sys

import tomlkit

START_COMMENT = f'<!-- {os.path.basename(__file__)}: start -->\n'
END_COMMENT = f'<!-- {os.path.basename(__file__)}: stop -->\n'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'openllm-core', 'src'))
from openllm_core.config import CONFIG_MAPPING


def markdown_noteblock(text: str):
  return ['\n', f'> **Note:** {text}\n']


def markdown_importantblock(text: str):
  return ['\n', f'> **Important:** {text}\n']


def main() -> int:
  with open(os.path.join(ROOT, 'openllm-python', 'pyproject.toml'), 'r') as f:
    deps = tomlkit.parse(f.read()).value['project']['optional-dependencies']
  with open(os.path.join(ROOT, 'README.md'), 'r') as f:
    readme = f.readlines()

  start_index, stop_index = readme.index(START_COMMENT), readme.index(END_COMMENT)

  content = []

  for it in CONFIG_MAPPING.values():
    it = it()
    architecture_name = it.__class__.__name__[:-6]
    details_block = ['<details>\n', f'<summary>{architecture_name}</summary>\n\n', '### Quickstart\n']
    if it['start_name'] in deps:
      instruction = f'> ```bash\n> pip install "openllm[{it["start_name"]}]"\n> ```'
      details_block.extend(markdown_noteblock(f'{architecture_name} requires to install with:\n{instruction}\n'))
    details_block.extend(
      [
        f'Run the following command to quickly spin up a {architecture_name} server:\n',
        f"""\
```bash
{'' if not it['trust_remote_code'] else 'TRUST_REMOTE_CODE=True '}openllm start {it['default_id']}
```""",
        'In a different terminal, run the following command to interact with the server:\n',
        """\
```bash
export OPENLLM_ENDPOINT=http://localhost:3000
openllm query 'What are large language models?'
```""",
        *markdown_noteblock(
          f"Any {architecture_name} variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search={it['model_name']}) to see more {architecture_name}-compatible models.\n"
        ),
        '\n### Supported models\n',
        f'You can specify any of the following {architecture_name} models via `openllm start`:\n\n',
      ]
    )
    list_ids = [f'- [{model_id}](https://huggingface.co/{model_id})' for model_id in it['model_ids']]
    details_block.extend(list_ids)
    details_block.extend(
      [
        '\n### Supported backends\n',
        'OpenLLM will support vLLM and PyTorch as default backend. By default, it will use vLLM if vLLM is available, otherwise fallback to PyTorch.\n',
        *markdown_importantblock(
          'We recommend user to explicitly specify `--backend` to choose the desired backend to run the model. If you have access to a GPU, always use `--backend vllm`.\n'
        ),
      ]
    )
    if 'vllm' in it['backend']:
      details_block.extend(
        [
          '\n- vLLM (Recommended):\n\n',
          'To install vLLM, run `pip install "openllm[vllm]"`\n',
          f"""\
```bash
{'' if not it['trust_remote_code'] else 'TRUST_REMOTE_CODE=True '}openllm start {it['model_ids'][0]} --backend vllm
```""",
          *markdown_importantblock(
            'Using vLLM requires a GPU that has architecture newer than 8.0 to get the best performance for serving. It is recommended that for all serving usecase in production, you should choose vLLM for serving.'
          ),
          *markdown_noteblock('Currently, adapters are yet to be supported with vLLM.'),
        ]
      )
    if 'pt' in it['backend']:
      details_block.extend(
        [
          '\n- PyTorch:\n\n',
          f"""\
```bash
{'' if not it['trust_remote_code'] else 'TRUST_REMOTE_CODE=True '}openllm start {it['model_ids'][0]} --backend pt
```""",
        ]
      )

    details_block.append('\n</details>\n\n')
    content.append('\n'.join(details_block))

  readme = readme[:start_index] + [START_COMMENT] + content + [END_COMMENT] + readme[stop_index + 1 :]
  with open(os.path.join(ROOT, 'README.md'), 'w') as f:
    f.writelines(readme)

  shutil.copyfile(os.path.join(ROOT, 'README.md'), os.path.join(ROOT, 'openllm-python', 'README.md'))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
