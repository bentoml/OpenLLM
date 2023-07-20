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
for ModelEnv construction and parsing environment variables.
"""
from __future__ import annotations
import contextlib
import logging
import os
import sys
import typing as t
from unittest import mock

import attr
import pytest
from hypothesis import assume
from hypothesis import given
from hypothesis import strategies as st

import openllm
import transformers
from openllm._configuration import GenerationConfig
from openllm._configuration import ModelSettings
from openllm._configuration import field_env_key

from ._strategies._configuration import make_llm_config
from ._strategies._configuration import model_settings


logger = logging.getLogger(__name__)

if t.TYPE_CHECKING:
    DictStrAny = dict[str, t.Any]
else:
    DictStrAny = dict


# XXX: @aarnphm fixes TypedDict behaviour in 3.11
@pytest.mark.skipif(
    sys.version_info[:2] == (3, 11), reason="TypedDict in 3.11 behaves differently, so we need to fix this"
)
def test_missing_default():
    with pytest.raises(ValueError, match="Missing required fields *"):
        make_llm_config("MissingDefaultId", {"name_type": "lowercase", "requirements": ["bentoml"]})
    with pytest.raises(ValueError, match="Missing required fields *"):
        make_llm_config("MissingModelId", {"default_id": "huggingface/t5-tiny-testing", "requirements": ["bentoml"]})
    with pytest.raises(ValueError, match="Missing required fields *"):
        make_llm_config(
            "MissingArchitecture",
            {
                "default_id": "huggingface/t5-tiny-testing",
                "model_ids": ["huggingface/t5-tiny-testing"],
                "requirements": ["bentoml"],
            },
        )


def test_forbidden_access():
    cl_ = make_llm_config(
        "ForbiddenAccess",
        {
            "default_id": "huggingface/t5-tiny-testing",
            "model_ids": ["huggingface/t5-tiny-testing", "bentoml/t5-tiny-testing"],
            "architecture": "PreTrainedModel",
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
    assert pytest.raises(
        openllm.exceptions.ForbiddenAttributeError,
        cl_.__getattribute__,
        cl_(),
        "SamplingParams",
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


@given(model_settings())
def test_config_derived_follow_attrs_protocol(gen_settings: ModelSettings):
    cl_ = make_llm_config("AttrsProtocolLLM", gen_settings)
    assert attr.has(cl_)


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
    assert sent.model_dump()["field1"] == field1
    assert sent.model_dump()["generation_config"]["temperature"] == temperature
    assert sent.model_dump(flatten=True)["field1"] == field1
    assert sent.model_dump(flatten=True)["temperature"] == temperature

    passed = cl_(field1=input_field1, temperature=input_temperature)
    assert passed.model_dump()["field1"] == input_field1
    assert passed.model_dump()["generation_config"]["temperature"] == input_temperature
    assert passed.model_dump(flatten=True)["field1"] == input_field1
    assert passed.model_dump(flatten=True)["temperature"] == input_temperature

    pas_nested = cl_(generation_config={"temperature": input_temperature}, field1=input_field1)
    assert pas_nested.model_dump()["field1"] == input_field1
    assert pas_nested.model_dump()["generation_config"]["temperature"] == input_temperature


@contextlib.contextmanager
def patch_env(**attrs: t.Any):
    with mock.patch.dict(os.environ, attrs, clear=True):
        yield


def test_struct_envvar():
    with patch_env(
        **{
            field_env_key("env_llm", "field1"): "4",
            field_env_key("env_llm", "temperature", suffix="generation"): "0.2",
        }
    ):

        class EnvLLM(openllm.LLMConfig):
            __config__ = {
                "default_id": "asdfasdf",
                "model_ids": ["asdf", "asdfasdfads"],
                "architecture": "PreTrainedModel",
            }
            field1: int = 2

            class GenerationConfig:
                temperature: float = 0.8

        sent = EnvLLM.model_construct_env()
        assert sent.field1 == 4
        assert sent["temperature"] == 0.2

        overwrite_default = EnvLLM()
        assert overwrite_default.field1 == 4
        assert overwrite_default["temperature"] == 0.2


def test_struct_provided_fields():
    class EnvLLM(openllm.LLMConfig):
        __config__ = {
            "default_id": "asdfasdf",
            "model_ids": ["asdf", "asdfasdfads"],
            "architecture": "PreTrainedModel",
        }
        field1: int = 2

        class GenerationConfig:
            temperature: float = 0.8

    sent = EnvLLM.model_construct_env(field1=20, temperature=0.4)
    assert sent.field1 == 20
    assert sent.generation_config.temperature == 0.4


def test_struct_envvar_with_overwrite_provided_env(monkeypatch: pytest.MonkeyPatch):
    with monkeypatch.context() as mk:
        mk.setenv(field_env_key("overwrite_with_env_available", "field1"), str(4.0))
        mk.setenv(field_env_key("overwrite_with_env_available", "temperature", suffix="generation"), str(0.2))
        sent = make_llm_config(
            "OverwriteWithEnvAvailable",
            {"default_id": "asdfasdf", "model_ids": ["asdf", "asdfasdfads"], "architecture": "PreTrainedModel"},
            fields=(("field1", "float", 3.0),),
        ).model_construct_env(field1=20.0, temperature=0.4)
        assert sent.generation_config.temperature == 0.4
        assert sent.field1 == 20.0


@given(model_settings())
@pytest.mark.parametrize(("return_dict", "typ"), [(True, DictStrAny), (False, transformers.GenerationConfig)])
def test_conversion_to_transformers(return_dict: bool, typ: type[t.Any], gen_settings: ModelSettings):
    cl_ = make_llm_config("ConversionLLM", gen_settings)
    assert isinstance(cl_().to_generation_config(return_as_dict=return_dict), typ)


@given(model_settings())
def test_click_conversion(gen_settings: ModelSettings):
    # currently our conversion omit Union type.
    def cli_mock(**attrs: t.Any):
        return attrs

    cl_ = make_llm_config("ClickConversionLLM", gen_settings)
    wrapped = cl_.to_click_options(cli_mock)
    filtered = {k for k, v in cl_.__openllm_hints__.items() if t.get_origin(v) is not t.Union}
    click_options_filtered = [i for i in wrapped.__click_params__ if i.name and not i.name.startswith("fake_")]
    assert len(filtered) == len(click_options_filtered)


@pytest.mark.parametrize("model_name", openllm.CONFIG_MAPPING.keys())
def test_configuration_dict_protocol(model_name: str):
    config = openllm.AutoConfig.for_model(model_name)
    assert isinstance(config.items(), list)
    assert isinstance(config.keys(), list)
    assert isinstance(config.values(), list)
    assert isinstance(dict(config), dict)
