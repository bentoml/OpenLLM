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
Types definition for OpenLLM.

Note that this module SHOULD NOT BE IMPORTED DURING RUNTIME, as this serve only for typing purposes.
It will raises a RuntimeError if this is imported eagerly.
"""
from typing import ParamSpec
from typing import TypeVar
from typing import cast

import click


if not TYPE_CHECKING:
    raise RuntimeError(f"{__name__} should not be imported during runtime")



P = ParamSpec("P")
O_co = TypeVar("O_co", covariant=True)


class ClickFunctionWrapper(Protocol[P, O_co]):
    __name__: str
    __click_params__: list[click.Option]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> O_co:
        ...


def is_click_function_wrapper(f: t.Callable[P, O_co]) -> bool:
    return (
        hasattr(f, "__name__")
        and hasattr(f, "__click_params__")
        and callable(f)
        and isinstance(f, t.Generic[P, O_co])
    )


def wrap_click_function(f: t.Callable[P, O_co]) -> ClickFunctionWrapper[P, O_co]:
    if not is_click_function_wrapper(f):
        raise TypeError(f"{f} is not a click function wrapper")

    return cast(ClickFunctionWrapper[P, O_co], f)
