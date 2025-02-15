# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jinja2",
#     "uv",
# ]
# ///

import subprocess, sys, pathlib, json

from jinja2 import Environment, FileSystemLoader

wd = pathlib.Path('.').parent
model_dict = subprocess.run(
    [sys.executable, '-m', 'uv', 'run', '--with-editable', '.', 'openllm', 'model', 'list', '--output', 'readme'],
    capture_output=True,
    text=True,
    check=True,
)
E = Environment(loader=FileSystemLoader('.'))
with (wd / 'README.md').open('w') as f:
    f.write(E.get_template('README.md.tpl').render(model_dict=json.loads(model_dict.stdout.strip())))
