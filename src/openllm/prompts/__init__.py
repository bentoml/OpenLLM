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
"""
Prompt interface for OpenLLM.

This module exposes the `Prompt` descriptor, which is used to create prompts as a ``bentoml.io.IODescriptor``.
This ``Prompt`` can also be used to interact with the client and provide similar interface to ``langchain.PromptTemplate``.

Example:
    >>> from openllm.prompts import Prompt
    >>> prompt = Prompt.from_template("Use the following as context: {context}!")
"""

from .descriptors import Prompt as Prompt
from .formatter import default_formatter as default_formatter
