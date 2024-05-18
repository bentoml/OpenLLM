#!/usr/bin/env bash
set -exuo pipefail

# Parent directory https://stackoverflow.com/a/246128/8643197
BASEDIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}"; )" &> /dev/null && pwd 2> /dev/null; )"

PIP_ARGS=()

# BentoML by default generates two requirement files:
#  - ./env/python/requirements.lock.txt: all dependencies locked to its version presented during `build`
#  - ./env/python/requirements.txt: all dependencies as user specified in code or requirements.txt file
REQUIREMENTS_TXT="$BASEDIR/requirements.txt"
REQUIREMENTS_LOCK="$BASEDIR/requirements.lock.txt"
WHEELS_DIR="$BASEDIR/wheels"
BENTOML_VERSION=${BENTOML_VERSION:-1.2.11}
# Install python packages, prefer installing the requirements.lock.txt file if it exist
pushd "$BASEDIR" &>/dev/null
if [ -f "$REQUIREMENTS_LOCK" ]; then
    echo "Installing pip packages from 'requirements.lock.txt'.."
    pip3 install -r "$REQUIREMENTS_LOCK" "${PIP_ARGS[@]}"
else
    if [ -f "$REQUIREMENTS_TXT" ]; then
        echo "Installing pip packages from 'requirements.txt'.."
        pip3 install -r "$REQUIREMENTS_TXT" "${PIP_ARGS[@]}"
    fi
fi
popd &>/dev/null

# Attempt to expand the glob pattern. The nullglob option ensures that
# the pattern itself is not returned if no files match.
shopt -s nullglob
wheels=($WHEELS_DIR/*.whl)

if [ ${#wheels[@]} -gt 0 ]; then
    echo "Installing wheels packaged in Bento.."
    pip3 install "${wheels[@]}" "${PIP_ARGS[@]}"
fi


# Install the BentoML from PyPI if it's not already installed
if python3 -c "import bentoml" &> /dev/null; then
    existing_bentoml_version=$(python3 -c "import bentoml; print(bentoml.__version__)")
    if [ "$existing_bentoml_version" != "$BENTOML_VERSION" ]; then
        echo "WARNING: using BentoML version ${existing_bentoml_version}"
    fi
else
    pip3 install bentoml=="$BENTOML_VERSION"
fi