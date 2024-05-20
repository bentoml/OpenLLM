import yaml
import sys
import subprocess
import os
import shutil
import tempfile
import pathlib


with open("recipe.yaml") as f:
    RECIPE = yaml.safe_load(f)


BENTOML_HOME = pathlib.Path(os.environ["BENTOML_HOME"])


CONSTANT_YAML_TMPL = r"""
CONSTANT_YAML = '''
{}
'''
"""


if __name__ == "__main__":
    if len(sys.argv) == 2:
        specified_model = sys.argv[1]
        if specified_model not in RECIPE:
            raise ValueError(f"Model {specified_model} not found in recipe")
    else:
        specified_model = None

    for model_name, config in RECIPE.items():
        if specified_model and model_name != specified_model:
            continue
        project = config["project"]
        model_repo, model_version = model_name.split(":")
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = pathlib.Path(tempdir)
            shutil.copytree(project, tempdir, dirs_exist_ok=True)

            with open(tempdir / "bento_constants.py", "w") as f:
                f.write(CONSTANT_YAML_TMPL.format(yaml.dump(config)))

            subprocess.run(
                ["bentoml", "build", str(tempdir), "--version", model_version],
                check=True,
                cwd=tempdir,
                env=os.environ,
            )

            # delete latest
            (BENTOML_HOME / "bentos" / model_repo / "latest").unlink(missing_ok=True)

            # link alias
            for alias in config.get("alias", []):
                if alias == "latest":
                    ALIAS_PATH = BENTOML_HOME / "bentos" / model_repo / alias
                    if ALIAS_PATH.exists():
                        continue
                    with open(ALIAS_PATH, "w") as f:
                        f.write(model_name)
                else:  # bentoml currently only support latest alias, copy to other alias
                    shutil.copytree(
                        BENTOML_HOME / "bentos" / model_repo / model_version,
                        BENTOML_HOME / "bentos" / model_repo / alias,
                    )
