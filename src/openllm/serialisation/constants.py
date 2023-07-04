# Copyright 2023 BentoML Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations


FRAMEWORK_TO_AUTOCLASS_MAPPING = {
    "pt": ("AutoModelForCausalLM", "AutoModelForSeq2SeqLM"),
    "tf": ("TFAutoModelForCausalLM", "TFAutoModelForSeq2SeqLM"),
    "flax": ("FlaxAutoModelForCausalLM", "FlaxAutoModelForSeq2SeqLM"),
}

# NOTE: This is a custom mapping for the autoclass to be used when loading as path
# since will try to infer the auto class to load, we will need this mapping
# in addition to FRAMEWORK_TO_AUTOCLASS_MAPPING for it to work properly.
# The following model all have trust_remote_code sets to True
MODEL_TO_AUTOCLASS_MAPPING = {
    "falcon": {"pt": "AutoModelForCausalLM"},
    "chatglm": {"pt": "AutoModel"},
    "mpt": {"pt": "AutoModelForCausalLM"},
}
