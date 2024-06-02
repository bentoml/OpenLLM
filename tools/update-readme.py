#!/usr/bin/env python3
import os, shutil, sys, tomlkit

START_COMMENT = f'<!-- {os.path.basename(__file__)}: start -->\n'
END_COMMENT = f'<!-- {os.path.basename(__file__)}: stop -->\n'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'openllm-core', 'src'))
from openllm_core.config import CONFIG_MAPPING
from openllm_core.config.configuration_auto import CONFIG_TO_ALIAS_NAMES


def markdown_noteblock(text: str):
  return ['\n', f'> **Note:** {text}\n']


def main() -> int:
  with open(os.path.join(ROOT, 'README.md'), 'r') as f:
    readme = f.readlines()

  start_index, stop_index = readme.index(START_COMMENT), readme.index(END_COMMENT)

  content = []

  for it in CONFIG_MAPPING.values():
    it = it()
    architecture_name = it.__class__.__name__[:-6]
    details_block = ['<details>\n', f'<summary>{architecture_name}</summary>\n\n', '### Quickstart\n']
    nitem = CONFIG_TO_ALIAS_NAMES[it.__class__.__name__]
    details_block.extend([
      f'Run the following command to quickly spin up a {architecture_name} server:\n',
      f"""\
```bash
openllm start {it['default_id']}{'' if not it['trust_remote_code'] else ' --trust-remote-code'}
```""",
      'You can run the following code in a different terminal to interact with the server:',
      """\
```python
import openllm_client
client = openllm_client.HTTPClient('http://localhost:3000')
client.generate('What are large language models?')
```""",
      *markdown_noteblock(
        f'Any {architecture_name} variants can be deployed with OpenLLM. Visit the [HuggingFace Model Hub](https://huggingface.co/models?sort=trending&search={nitem}) to see more {architecture_name}-compatible models.\n'
      ),
      '\n### Supported models\n',
      f'You can specify any of the following {architecture_name} models via `openllm start`:\n\n',
    ])
    list_ids = [f'- [{model_id}](https://huggingface.co/{model_id})' for model_id in it['model_ids']]
    details_block.extend(list_ids)
    details_block.append('\n</details>\n\n')

    content.append('\n'.join(details_block))

  readme = readme[:start_index] + [START_COMMENT] + content + [END_COMMENT] + readme[stop_index + 1 :]
  with open(os.path.join(ROOT, 'README.md'), 'w') as f:
    f.writelines(readme)

  shutil.copyfile(os.path.join(ROOT, 'README.md'), os.path.join(ROOT, 'openllm-python', 'README.md'))
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
