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

import typing as t

from .utils import is_flax_available
from .utils import is_tf_available
from .utils import is_torch_available


try:
    import pytest
except ImportError:
    raise ImportError("You need to install pytest to use 'openllm.tests' utilities: 'pip install pytest'")


def require_tf(f: t.Callable[..., t.Any]):
    return pytest.mark.skipif(not is_tf_available(), reason="requires TensorFlow")(f)


def require_flax(f: t.Callable[..., t.Any]):
    return pytest.mark.skipif(not is_flax_available(), reason="requires Flax")(f)


def require_torch(f: t.Callable[..., t.Any]):
    return pytest.mark.skipif(not is_torch_available(), reason="requires PyTorch")(f)
