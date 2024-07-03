import hashlib
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

import yaml

with open("recipe.yaml") as f:
    RECIPE = yaml.safe_load(f)


BENTOML_HOME = pathlib.Path(os.environ["BENTOML_HOME"])


CONSTANT_YAML_TMPL = r"""
CONSTANT_YAML = '''
{}
'''
"""


def hash_file(file_path):
    """计算单个文件的哈希值"""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_directory(directory_path):
    """计算文件夹内容的哈希值"""
    hasher = hashlib.sha256()

    for root, _, files in sorted(os.walk(directory_path)):
        for file in sorted(files):
            file_path = os.path.join(root, file)
            file_hash = hash_file(file_path)
            hasher.update(file_hash.encode())

    return hasher.hexdigest()


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

            labels = config.get("extra_labels", {})
            yaml_content = yaml.load(
                (tempdir / "bentofile.yaml").read_text(), Loader=yaml.SafeLoader
            )
            with open(tempdir / "bentofile.yaml", "w") as f:
                yaml_content["labels"] = dict(yaml_content.get("labels", {}), **labels)
                f.write(yaml.dump(yaml_content))

            directory_hash = hash_directory(tempdir)
            model_version = f"{model_version}-{directory_hash[:4]}"

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
                        f.write(model_version)
                else:  # bentoml currently only support latest alias, copy to other alias
                    shutil.copytree(
                        BENTOML_HOME / "bentos" / model_repo / model_version,
                        BENTOML_HOME / "bentos" / model_repo / alias,
                    )
