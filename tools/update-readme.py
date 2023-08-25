#!/usr/bin/env python3
from __future__ import annotations
import os, inflection, tomlkit, sys
import typing as t
START_COMMENT = f'<!-- {os.path.basename(__file__)}: start -->\n'
END_COMMENT = f'<!-- {os.path.basename(__file__)}: stop -->\n'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'openllm-python', 'src'))
import openllm
def main() -> int:
  with open(os.path.join(ROOT, 'openllm-python', 'pyproject.toml'), 'r') as f:
    deps = tomlkit.parse(f.read()).value['project']['optional-dependencies']
  with open(os.path.join(ROOT, 'README.md'), 'r') as f:
    readme = f.readlines()

  start_index, stop_index = readme.index(START_COMMENT), readme.index(END_COMMENT)
  formatted: dict[t.Literal['Model', 'Architecture', 'URL', 'Installation', 'Model Ids'], list[str | list[str]]] = {
      'Model': [], 'Architecture': [], 'URL': [], 'Model Ids': [], 'Installation': [],
  }
  max_install_len_div = 0
  for name, config_cls in openllm.CONFIG_MAPPING.items():
    dashed = inflection.dasherize(name)
    formatted['Model'].append(dashed)
    formatted['Architecture'].append(config_cls.__openllm_architecture__)
    formatted['URL'].append(config_cls.__openllm_url__)
    formatted['Model Ids'].append(config_cls.__openllm_model_ids__)
    if dashed in deps: instruction = f'```bash\npip install "openllm[{dashed}]"\n```'
    else: instruction = '```bash\npip install openllm\n```'
    if len(instruction) > max_install_len_div: max_install_len_div = len(instruction)
    formatted['Installation'].append(instruction)
  meta: list[str] = ['\n', "<table align='center'>\n"]

  # NOTE: headers
  meta += ['<tr>\n']
  meta.extend([f'<th>{header}</th>\n' for header in formatted.keys() if header not in ('URL',)])
  meta += ['</tr>\n']
  # NOTE: rows
  for name, architecture, url, model_ids, installation in t.cast(t.Iterable[t.Tuple[str, str, str, t.List[str], str]], zip(*formatted.values())):
    meta += '<tr>\n'
    # configure architecture URL
    cfg_cls = openllm.CONFIG_MAPPING[name]
    if cfg_cls.__openllm_trust_remote_code__: arch = f'<td><a href={url}><code>{architecture}</code></a></td>\n'
    else:
      arch = f"<td><a href=https://huggingface.co/docs/transformers/main/model_doc/{dict(dolly_v2='gpt_neox',stablelm='gpt_neox', starcoder='gpt_bigcode', flan_t5='t5').get(cfg_cls.__openllm_model_name__, cfg_cls.__openllm_model_name__)}#transformers.{architecture}><code>{architecture}</code></a></td>\n"
    meta.extend([f'\n<td><a href={url}>{name}</a></td>\n', arch])
    format_with_links: list[str] = []
    for lid in model_ids:
      format_with_links.append(f'<li><a href=https://huggingface.co/{lid}><code>{lid}</code></a></li>')
    meta.append('<td>\n\n<ul>' + '\n'.join(format_with_links) + '</ul>\n\n</td>\n')
    meta.append(f'<td>\n\n{installation}\n\n</td>\n')
    meta += '</tr>\n'
  meta.extend(['</table>\n', '\n'])

  readme = readme[:start_index] + [START_COMMENT] + meta + [END_COMMENT] + readme[stop_index + 1:]
  with open(os.path.join(ROOT, 'README.md'), 'w') as f:
    f.writelines(readme)
  return 0
if __name__ == '__main__': raise SystemExit(main())
