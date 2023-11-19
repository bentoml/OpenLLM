#!/usr/bin/env bash

printf "Running mirror.sh\n"
bash ./tools/mirror.sh
printf "Running update-mypy.py\n"
python ./tools/update-mypy.py
printf "Running update-config-stubs.py\n"
python ./tools/dependencies.py
printf "Running dependencies.py\n"
python ./tools/update-config-stubs.py
