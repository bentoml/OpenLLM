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

from .utils import LazyLoader
from .utils import is_bitsandbytes_available
from .utils import is_transformers_supports_kbit
from .utils import pkg


if t.TYPE_CHECKING:
    import torch

    import openllm
    import transformers

    from ._types import DictStrAny
else:
    torch = LazyLoader("torch", globals(), "torch")
    transformers = LazyLoader("transformers", globals(), "transformers")

logger = logging.getLogger(__name__)

QuantiseMode = t.Literal["int8", "int4", "gptq"]


def infer_quantisation_config(
    cls: type[openllm.LLM[t.Any, t.Any]], quantise: QuantiseMode, **attrs: t.Any
) -> tuple[transformers.BitsAndBytesConfig | t.Any, DictStrAny]:
    # 8 bit configuration
    int8_threshold = attrs.pop("llm_int8_threshhold", 6.0)
    int8_enable_fp32_cpu_offload = attrs.pop("llm_int8_enable_fp32_cpu_offload", False)
    int8_skip_modules: list[str] | None = attrs.pop("llm_int8_skip_modules", None)
    int8_has_fp16_weight = attrs.pop("llm_int8_has_fp16_weight", False)

    def create_int8_config(int8_skip_modules: list[str] | None):
        if int8_skip_modules is None:
            int8_skip_modules = []
        if "lm_head" not in int8_skip_modules and cls.config_class.__openllm_model_type__ == "causal_lm":
            logger.debug("Skipping 'lm_head' for quantization for %s", cls.__name__)
            int8_skip_modules.append("lm_head")
        return transformers.BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=int8_enable_fp32_cpu_offload,
            llm_int8_threshhold=int8_threshold,
            llm_int8_skip_modules=int8_skip_modules,
            llm_int8_has_fp16_weight=int8_has_fp16_weight,
        )

    # 4 bit configuration
    int4_compute_dtype = attrs.pop("bnb_4bit_compute_dtype", torch.bfloat16)
    int4_quant_type = attrs.pop("bnb_4bit_quant_type", "nf4")
    int4_use_double_quant = attrs.pop("bnb_4bit_use_double_quant", True)

    # NOTE: Quantization setup
    # quantize is a openllm.LLM feature, where we can quantize the model
    # with bitsandbytes or quantization aware training.
    if not is_bitsandbytes_available():
        raise RuntimeError(
            "Quantization requires bitsandbytes to be installed. Make "
            "sure to install OpenLLM with 'pip install \"openllm[fine-tune]\"'"
        )
    if quantise == "int8":
        quantisation_config = create_int8_config(int8_skip_modules)
    elif quantise == "int4":
        if is_transformers_supports_kbit():
            quantisation_config = transformers.BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=int4_compute_dtype,
                bnb_4bit_quant_type=int4_quant_type,
                bnb_4bit_use_double_quant=int4_use_double_quant,
            )
        else:
            logger.warning(
                "'quantize' is set to int4, while the current transformers version %s does not support "
                "k-bit quantization. k-bit quantization is supported since transformers 4.30, therefore "
                "make sure to install the latest version of transformers either via PyPI or "
                "from git source: 'pip install git+https://github.com/huggingface/transformers'.",
                pkg.pkg_version_info("transformers"),
            )
            logger.warning("OpenLLM will fallback to 8-bit quantization.")
            quantisation_config = create_int8_config(int8_skip_modules)
    elif quantise == "gptq":
        # TODO: support GPTQ loading quantization
        raise NotImplementedError("GPTQ is not supported yet.")
    else:
        raise ValueError(f"'quantize' must be one of ['int8', 'int4', 'gptq'], got {quantise} instead.")

    return quantisation_config, attrs
