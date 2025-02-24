# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "jinja2",
#     "uv",
# ]
# ///
import subprocess, sys, pathlib, json, jinja2

with (pathlib.Path('.').parent / 'README.md').open('w') as f:
    f.write(
        jinja2.Environment(loader=jinja2.FileSystemLoader('.'))
        .get_template('README.md.tpl')
        .render(
            model_dict=json.loads(
                subprocess.run(
                    [
                        sys.executable,
                        '-m',
                        'uv',
                        'run',
                        '--with-editable',
                        '.',
                        'openllm',
                        'model',
                        'list',
                        '--output',
                        'readme',
                    ],
                    text=True,
                    check=True,
                    capture_output=True,
                ).stdout.strip()
            )
        )
    )
