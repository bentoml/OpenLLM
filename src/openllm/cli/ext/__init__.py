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

"""OpenLLM CLI Extension.

The following directory contains all possible extensions for OpenLLM CLI
For adding new extension, just simply name that ext to `<name_ext>.py` and define
a ``click.command()`` with the following format:

```python
import click

@click.command(<name_ext>)
...
def cli(...): # <- this is important here, it should always name CLI in order for the extension resolver to know how to import this extensions.
```

NOTE: Make sure to keep this file blank such that it won't mess with the import order.
"""
