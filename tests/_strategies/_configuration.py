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
import logging
import typing as t

from hypothesis import strategies as st

import openllm
from openllm._configuration import ModelSettings


logger = logging.getLogger(__name__)

env_strats = st.sampled_from([openllm.utils.EnvVarMixin(model_name) for model_name in openllm.CONFIG_MAPPING.keys()])


@st.composite
def model_settings(draw: st.DrawFn):
    """Strategy for generating ModelSettings objects."""
    kwargs: dict[str, t.Any] = {
        "default_id": st.text(min_size=1),
        "model_ids": st.lists(st.text(), min_size=1),
        "architecture": st.text(min_size=1),
        "url": st.text(),
        "requires_gpu": st.booleans(),
        "trust_remote_code": st.booleans(),
        "requirements": st.none() | st.lists(st.text(), min_size=1),
        "default_implementation": st.dictionaries(st.sampled_from(["cpu", "nvidia.com/gpu"]), st.sampled_from(["vllm", "pt", "tf", "flax"])),
        "model_type": st.sampled_from(["causal_lm", "seq2seq_lm"]),
        "runtime": st.sampled_from(["transformers", "ggml"]),
        "name_type": st.sampled_from(["dasherize", "lowercase"]),
        "timeout": st.integers(min_value=3600),
        "workers_per_resource": st.one_of(st.integers(min_value=1), st.floats(min_value=0.1, max_value=1.0)),
    }
    return draw(st.builds(ModelSettings, **kwargs))


def make_llm_config(
    cls_name: str,
    dunder_config: dict[str, t.Any] | ModelSettings,
    fields: tuple[tuple[t.LiteralString, str, t.Any], ...] | None = None,
    generation_fields: tuple[tuple[t.LiteralString, t.Any], ...] | None = None,
) -> type[openllm.LLMConfig]:
    globs: dict[str, t.Any] = {"openllm": openllm}
    _config_args: list[str] = []
    lines: list[str] = [f"class {cls_name}Config(openllm.LLMConfig):"]
    for attr, value in dunder_config.items():
        _config_args.append(f'"{attr}": __attr_{attr}')
        globs[f"_{cls_name}Config__attr_{attr}"] = value
    lines.append(f'    __config__ = {{ {", ".join(_config_args)} }}')
    if fields is not None:
        for field, type_, default in fields:
            lines.append(f"    {field}: {type_} = openllm.LLMConfig.Field({default!r})")
    if generation_fields is not None:
        generation_lines = ["class GenerationConfig:"]
        for field, default in generation_fields:
            generation_lines.append(f"    {field} = {default!r}")
        lines.extend(("    " + line for line in generation_lines))

    script = "\n".join(lines)

    if openllm.utils.DEBUG:
        logger.info("Generated class %s:\n%s", cls_name, script)

    eval(compile(script, "name", "exec"), globs)

    return globs[f"{cls_name}Config"]
