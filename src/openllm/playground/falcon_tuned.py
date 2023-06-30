from __future__ import annotations

import dataclasses
import logging
import os
import sys
import typing as t

# import openllm here for OPENLLMDEVDEBUG
import openllm

openllm.utils.configure_logging()

logger = logging.getLogger(__name__)

if len(openllm.utils.gpu_count()) < 1:
    raise RuntimeError("This script can only be run with system that GPU is available.")

_deps = ["trl", '"openllm[fine-tune]"']

if openllm.utils.DEBUG:
    logger.info("Installing dependencies to run this script: %s", _deps)

    if os.system(f"pip install -U {' '.join(_deps)}") != 0:
        raise SystemExit(1)

from datasets import load_dataset
from peft import LoraConfig
from peft import get_peft_model
from trl import SFTTrainer
