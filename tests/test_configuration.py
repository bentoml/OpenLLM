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

"""All configuration-related tests for openllm.LLMConfig. This will include testing
for ModelEnv construction and parsing environment variables."""
from __future__ import annotations

import logging

import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import strategies as st

import openllm
from openllm._configuration import GenerationConfig
from openllm._configuration import ModelSettings
from openllm._configuration import _field_env_key
from openllm.utils import DEBUG

from ._strategies._configuration import make_llm_config
from ._strategies._configuration import model_settings


logger = logging.getLogger(__name__)


def test_missing_default():
    with pytest.raises(ValueError, match="Either 'default_id' or 'model_ids'*"):
        make_llm_config("MissingDefaultId", {"name_type": "lowercase", "requirements": ["bentoml"]})

    with pytest.raises(ValueError, match="Either 'default_id' or 'model_ids'*"):
        make_llm_config("MissingModelId", {"default_id": "huggingface/t5-tiny-testing", "requirements": ["bentoml"]})


def test_forbidden_access():
    cl_ = make_llm_config(
        "ForbiddenAccess",
        {
            "default_id": "huggingface/t5-tiny-testing",
            "model_ids": ["huggingface/t5-tiny-testing", "bentoml/t5-tiny-testing"],
            "requirements": ["bentoml"],
        },
    )

    assert pytest.raises(
        openllm.exceptions.ForbiddenAttributeError,
        cl_.__getattribute__,
        cl_(),
        "__config__",
    )
    assert pytest.raises(
        openllm.exceptions.ForbiddenAttributeError,
        cl_.__getattribute__,
        cl_(),
        "GenerationConfig",
    )

    assert openllm.utils.lenient_issubclass(cl_.__openllm_generation_class__, GenerationConfig)


@given(model_settings())
def test_class_normal_gen(gen_settings: ModelSettings):
    assume(gen_settings["default_id"] and all(i for i in gen_settings["model_ids"]))
    cl_: type[openllm.LLMConfig] = make_llm_config("NotFullLLM", gen_settings)
    assert issubclass(cl_, openllm.LLMConfig)
    for key in gen_settings:
        assert object.__getattribute__(cl_, f"__openllm_{key}__") == gen_settings.__getitem__(key)


@given(model_settings(), st.integers())
def test_simple_struct_dump(gen_settings: ModelSettings, field1: int):
    cl_ = make_llm_config("IdempotentLLM", gen_settings, fields=(("field1", "float", field1),))
    assert cl_().model_dump()["field1"] == field1


@given(model_settings(), st.integers())
def test_config_derivation(gen_settings: ModelSettings, field1: int):
    cl_ = make_llm_config("IdempotentLLM", gen_settings, fields=(("field1", "float", field1),))
    new_cls = cl_.model_derivate("DerivedLLM", default_id="asdfasdf")
    assert new_cls.__openllm_default_id__ == "asdfasdf"


@given(
    model_settings(),
    st.integers(max_value=283473),
    st.floats(min_value=0.0, max_value=1.0),
    st.integers(max_value=283473),
    st.floats(min_value=0.0, max_value=1.0),
)
def test_complex_struct_dump(
    gen_settings: ModelSettings, field1: int, temperature: float, input_field1: int, input_temperature: float
):
    cl_ = make_llm_config(
        "ComplexLLM",
        gen_settings,
        fields=(("field1", "float", field1),),
        generation_fields=(("temperature", temperature),),
    )
    sent = cl_()
    assert (
        sent.model_dump()["field1"] == field1 and sent.model_dump()["generation_config"]["temperature"] == temperature
    )
    assert (
        sent.model_dump(flatten=True)["field1"] == field1
        and sent.model_dump(flatten=True)["temperature"] == temperature
    )

    passed = cl_(field1=input_field1, temperature=input_temperature)
    assert (
        passed.model_dump()["field1"] == input_field1
        and passed.model_dump()["generation_config"]["temperature"] == input_temperature
    )
    assert (
        passed.model_dump(flatten=True)["field1"] == input_field1
        and passed.model_dump(flatten=True)["temperature"] == input_temperature
    )

    pas_nested = cl_(generation_config={"temperature": input_temperature}, field1=input_field1)
    assert (
        pas_nested.model_dump()["field1"] == input_field1
        and pas_nested.model_dump()["generation_config"]["temperature"] == input_temperature
    )


def test_struct_envvar(monkeypatch: pytest.MonkeyPatch):
    class EnvLLM(openllm.LLMConfig):
        __config__ = {"default_id": "asdfasdf", "model_ids": ["asdf", "asdfasdfads"]}
        field1: int = 2

        class GenerationConfig:
            temperature: float = 0.8

    f1_env = _field_env_key(EnvLLM.__openllm_model_name__, "field1")
    temperature_env = _field_env_key(EnvLLM.__openllm_model_name__, "temperature", suffix="generation")

    if DEBUG:
        logger.info(f"Env keys: {f1_env}, {temperature_env}")

    with monkeypatch.context() as m:
        m.setenv(f1_env, "4")
        m.setenv(temperature_env, "0.2")
        sent = EnvLLM()
        assert sent.field1 == 4
        assert sent.generation_config.temperature == 0.8

    # NOTE: This is the expected behaviour, where users pass in value, we respect it over envvar.
    with monkeypatch.context() as m:
        m.setenv(f1_env, "4")
        m.setenv(temperature_env, "0.2")
        sent = EnvLLM.model_construct_env(field1=20, temperature=0.4)
        assert sent.field1 == 4
        assert sent.generation_config.temperature == 0.4
