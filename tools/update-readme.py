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
  return [
    f"""<div class="markdown-alert markdown-alert-note" dir="auto"><p class="markdown-alert-title" dir="auto"><svg class="octicon octicon-info mr-2" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8Zm8-6.5a6.5 6.5 0 1 0 0 13 6.5 6.5 0 0 0 0-13ZM6.5 7.75A.75.75 0 0 1 7.25 7h1a.75.75 0 0 1 .75.75v2.75h.25a.75.75 0 0 1 0 1.5h-2a.75.75 0 0 1 0-1.5h.25v-2h-.25a.75.75 0 0 1-.75-.75ZM8 6a1 1 0 1 1 0-2 1 1 0 0 1 0 2Z"></path></svg>Note</p><p dir="auto">{text}</p></div>"""
  ]


def markdown_importantblock(text: str):
  return [
    f"""<div class="markdown-alert markdown-alert-important" dir="auto"><p class="markdown-alert-title" dir="auto"><svg class="octicon octicon-report mr-2" viewBox="0 0 16 16" version="1.1" width="16" height="16" aria-hidden="true"><path d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v9.5A1.75 1.75 0 0 1 14.25 13H8.06l-2.573 2.573A1.458 1.458 0 0 1 3 14.543V13H1.75A1.75 1.75 0 0 1 0 11.25Zm1.75-.25a.25.25 0 0 0-.25.25v9.5c0 .138.112.25.25.25h2a.75.75 0 0 1 .75.75v2.19l2.72-2.72a.749.749 0 0 1 .53-.22h6.5a.25.25 0 0 0 .25-.25v-9.5a.25.25 0 0 0-.25-.25Zm7 2.25v2.5a.75.75 0 0 1-1.5 0v-2.5a.75.75 0 0 1 1.5 0ZM9 9a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z"></path></svg>Important</p><p dir="auto">{text}</p></div>"""
  ]


def main() -> int:
  with open(os.path.join(ROOT, 'openllm-python', 'pyproject.toml'), 'r') as f:
    deps = tomlkit.parse(f.read()).value['project']['optional-dependencies']
  with open(os.path.join(ROOT, 'README.md'), 'r') as f:
    readme = f.readlines()

  start_index, stop_index = readme.index(START_COMMENT), readme.index(END_COMMENT)

  content = []

  for it in CONFIG_MAPPING.values():
    it = it()
    details_block = ['<details>\n']
    architecture_name = it.__class__.__name__[:-6]
    details_block.extend(
      [
        f'<summary>{architecture_name}</summary>\n\n',
        '### Quickstart\n',
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
        '### Supported models\n',
        f'You can specify any of the following {architecture_name} models via `openllm start`:\n\n',
      ]
    )
    list_ids = [f'- [{model_id}](https://huggingface.co/{model_id})' for model_id in it['model_ids']]
    details_block.extend(list_ids)
    details_block.extend(
      [
        '### Supported backends\n',
        'OpenLLM will support vLLM and PyTorch as default backend. By default, it will use vLLM if vLLM is available, otherwise fallback to PyTorch.\n',
        *markdown_importantblock(
          'We recommend user to explicitly specify <code>--backend</code> to choose the desired backend to run the model. If you have access to a GPU, always use <code>--backend vllm</code>.\n'
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
openllm start {it['model_ids'][0]} --backend vllm
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
openllm start {it['model_ids'][0]} --backend pt
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
