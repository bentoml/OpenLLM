# Changelog

We are following [semantic versioning](https://semver.org/) with strict
backward-compatibility policy.

You can find out backwards-compatibility policy
[here](https://github.com/bentoml/openllm/blob/main/.github/SECURITY.md).

Changes for the upcoming release can be found in the
['changelog.d' directory](https://github.com/bentoml/openllm/tree/main/changelog.d)
in our repository.

<!--
Do *NOT* add changelog entries here!

This changelog is managed by towncrier and is compiled at release time.
-->

<!-- towncrier release notes start -->

## [0.2.6](https://github.com/bentoml/openllm/tree/v0.2.6)

### Backwards-incompatible Changes

- Updated signature for `load_model` and `load_tokenizer` not to allow tag.
  Tag can be accessed via `llm.tag`, or if using `openllm.serialisation` or `bentoml.transformers` then you can use `self._bentomodel`

  Updated serialisation shared logics to reduce callstack for saving three calltrace.
  [#132](https://github.com/bentoml/openllm/issues/132)


## [0.2.5](https://github.com/bentoml/openllm/tree/v0.2.5)

### Features

- Added support for sending arguments via CLI.

  ```python
  openllm query --endpoint localhost:3000 "What is the difference between noun and pronoun?" --sampling-params temperature 0.84
  ```

  Fixed llama2 qlora training script to save unquantized weights
  [#130](https://github.com/bentoml/openllm/issues/130)


## [0.2.4](https://github.com/bentoml/openllm/tree/v0.2.4)
No significant changes.


## [0.2.3](https://github.com/bentoml/openllm/tree/v0.2.3)
No significant changes.


## [0.2.2](https://github.com/bentoml/openllm/tree/v0.2.2)
No significant changes.


## [0.2.1](https://github.com/bentoml/openllm/tree/v0.2.1)
No significant changes.


## [0.2.0](https://github.com/bentoml/openllm/tree/v0.2.0)

### Features

- Added support for GPTNeoX models. All variants of GPTNeoX, including Dolly-V2
  and StableLM can now also use `openllm start gpt-neox`

  `openllm models -o json` nows return CPU and GPU field. `openllm models` now
  show table that mimics the one from README.md

  Added scripts to automatically add models import to `__init__.py`

  `--workers-per-resource` now accepts the following strategies:

  - `round_robin`: Similar behaviour when setting `--workers-per-resource 1`. This
    is useful for smaller models.
  - `conserved`: This will determine the number of available GPU resources, and
    only assign one worker for the LLMRunner with all available GPU resources. For
    example, if ther are 4 GPUs available, then `conserved` is equivalent to
    `--workers-per-resource 0.25`.
  [#106](https://github.com/bentoml/openllm/issues/106)
- Added support for [Baichuan](https://github.com/baichuan-inc/Baichuan-7B) model
  generation, contributed by @hetaoBackend

  Fixes how we handle model loader auto class for trust_remote_code in
  transformers
  [#115](https://github.com/bentoml/openllm/issues/115)


### Bug fix

- Fixes relative model_id handling for running LLM within the container.

  Added support for building container directly with `openllm build`. Users now
  can do `openllm build --format=container`:

  ```bash
  openllm build flan-t5 --format=container
  ```

  This is equivalent to:

  ```bash
  openllm build flan-t5 && bentoml containerize google-flan-t5-large-service
  ```

  Added Snapshot testing and more robust edge cases for model testing

  General improvement in `openllm.LLM.import_model` where it will parse santised
  parameters automatically.

  Fixes `openllm start <bento>` to use correct `model_id`, ignoring `--model-id`
  (The correct behaviour)

  Fixes `--workers-per-resource conserved` to respect `--device`

  Added initial interface for `LLM.embeddings`
  [#107](https://github.com/bentoml/openllm/issues/107)
- Fixes resources to correctly follows CUDA_VISIBLE_DEVICES spec

  OpenLLM now contains a standalone parser that mimic `torch.cuda` parser for set
  GPU devices. This parser will be used to parse both AMD and NVIDIA GPUs.

  `openllm` should now be able to parse `GPU-` and `MIG-` UUID from both
  configuration or spec.
  [#114](https://github.com/bentoml/openllm/issues/114)


## [0.1.20](https://github.com/bentoml/openllm/tree/v0.1.20)

### Features

- ### Fine-tuning support for Falcon

  Added support for fine-tuning Falcon models with QLoRa

  OpenLLM now brings a `openllm playground`, which create a jupyter notebook for
  easy fine-tuning script

  Currently, it supports fine-tuning OPT and Falcon, more to come.

  `openllm.LLM` now provides a `prepare_for_training` helpers to easily setup LoRA
  and related configuration for fine-tuning
  [#98](https://github.com/bentoml/openllm/issues/98)


### Bug fix

- Fixes loading MPT config on CPU

  Fixes runner StopIteration on GET for Starlette App
  [#92](https://github.com/bentoml/openllm/issues/92)
- `openllm.LLM` now generates tags based on given `model_id` and optional
  `model_version`.

  If given `model_id` is a custom path, the name would be the basename of the
  directory, and version would be the hash of the last modified time.

  `openllm start` now provides a `--runtime`, allowing setup different runtime.
  Currently it refactors to `transformers`. GGML is working in progress.

  Fixes miscellaneous items when saving models with quantized weights.
  [#102](https://github.com/bentoml/openllm/issues/102)


## [0.1.19](https://github.com/bentoml/openllm/tree/v0.1.19)
No significant changes.


## [0.1.18](https://github.com/bentoml/openllm/tree/v0.1.18)

### Features

- `openllm.LLMConfig` now supports `dict()` protocol

  ```bash

  config = openllm.LLMConfig.for_model("opt")

  print(config.items())
  print(config.values())
  print(config.keys())
  print(dict(config))
  ```
  [#85](https://github.com/bentoml/openllm/issues/85)
- Added supports for MPT to OpenLLM

  Fixes a LLMConfig to only parse environment when it is available
  [#91](https://github.com/bentoml/openllm/issues/91)


## [0.1.17](https://github.com/bentoml/openllm/tree/v0.1.17)

### Bug fix

- Fixes loading logics from custom path. If given model path are given, OpenLLM
  won't try to import it to the local store.

  OpenLLM now only imports and fixes the models to loaded correctly within the
  bento, see the generated service for more information.

  Fixes service not ready when serving within a container or on BentoCloud. This
  has to do with how we load the model before in the bento.

  Falcon loading logics has been reimplemented to fix this major bug. Make sure to
  delete all previous save weight for falcon with `openllm prune`

  `openllm start` now supports bento

  ```bash
  openllm start llm-bento --help
  ```
  [#80](https://github.com/bentoml/openllm/issues/80)


## [0.1.16](https://github.com/bentoml/openllm/tree/v0.1.16)
No significant changes.


## [0.1.15](https://github.com/bentoml/openllm/tree/v0.1.15)

### Features

- `openllm.Runner` now supports AMD GPU, addresses #65.

  It also respect CUDA_VISIBLE_DEVICES set correctly, allowing disabling GPU and
  run on CPU only.
  [#72](https://github.com/bentoml/openllm/issues/72)


## [0.1.14](https://github.com/bentoml/openllm/tree/v0.1.14)

### Features

- Added support for standalone binary distribution. Currently works on Linux and
  Windows:

  The following are supported:

  - aarch64-unknown-linux-gnu
  - x86_64-unknown-linux-gnu
  - x86_64-unknown-linux-musl
  - i686-unknown-linux-gnu
  - powerpc64le-unknown-linux-gnu
  - x86_64-pc-windows-msvc
  - i686-pc-windows-msvc

  Reverted matrices expansion for CI to all Python version. Now leveraging Hatch
  env matrices
  [#66](https://github.com/bentoml/openllm/issues/66)


### Bug fix

- Moved implementation of dolly-v2 and falcon serialization to save PreTrainedModel instead of pipeline.

  Save dolly-v2 now save the actual model instead of the pipeline abstraction. If you have a Dolly-V2
  model available locally, kindly ask you to do `openllm prune` to have the new implementation available.

  Dolly-v2 and falcon nows implements some memory optimization to help with loading with lower resources system

  Configuration removed field: 'use_pipeline'
  [#60](https://github.com/bentoml/openllm/issues/60)
- Remove duplicated class instance of `generation_config` as it should be set via
  instance attributes.

  fixes tests flakiness and one broken cases for parsing env
  [#64](https://github.com/bentoml/openllm/issues/64)


## [0.1.13](https://github.com/bentoml/openllm/tree/v0.1.13)
No significant changes.


## [0.1.12](https://github.com/bentoml/openllm/tree/v0.1.12)

### Features

- Serving LLM with fine-tuned LoRA, QLoRA adapters layers

  Then the given fine tuning weights can be served with the model via
  `openllm start`:

  ```bash
  openllm start opt --model-id facebook/opt-6.7b --adapter-id /path/to/adapters
  ```

  If you just wish to try some pretrained adapter checkpoint, you can use
  `--adapter-id`:

  ```bash
  openllm start opt --model-id facebook/opt-6.7b --adapter-id aarnphm/opt-6.7b-lora
  ```

  To use multiple adapters, use the following format:

  ```bash
  openllm start opt --model-id facebook/opt-6.7b --adapter-id aarnphm/opt-6.7b-lora --adapter-id aarnphm/opt-6.7b-lora:french_lora
  ```

  By default, the first `adapter-id` will be the default lora layer, but
  optionally users can change what lora layer to use for inference via
  `/v1/adapters`:

  ```bash
  curl -X POST http://localhost:3000/v1/adapters --json '{"adapter_name": "vn_lora"}'
  ```

  > Note that for multiple `adapter-name` and `adapter-id`, it is recomended to
  > update to use the default adapter before sending the inference, to avoid any
  > performance degradation

  To include this into the Bento, one can also provide a `--adapter-id` into
  `openllm build`:

  ```bash
  openllm build opt --model-id facebook/opt-6.7b --adapter-id ...
  ```

  Separate out configuration builder, to make it more flexible for future
  configuration generation.
  [#52](https://github.com/bentoml/openllm/issues/52)


### Bug fix

- Fixes how `llm.ensure_model_id_exists` parse `openllm download` correctly

  Renamed `openllm.utils.ModelEnv` to `openllm.utils.EnvVarMixin`
  [#58](https://github.com/bentoml/openllm/issues/58)


## [0.1.11](https://github.com/bentoml/openllm/tree/v0.1.11)
No significant changes.


## [0.1.10](https://github.com/bentoml/openllm/tree/v0.1.10)
No significant changes.


## [0.1.9](https://github.com/bentoml/openllm/tree/v0.1.9)

### Changes

- Fixes setting logs for agents to info instead of logger object.
  [#37](https://github.com/bentoml/openllm/issues/37)


## [0.1.8](https://github.com/bentoml/openllm/tree/v0.1.8)
No significant changes.


## [0.1.7](https://github.com/bentoml/openllm/tree/v0.1.7)

### Features

- OpenLLM now seamlessly integrates with HuggingFace Agents.
  Replace the HfAgent endpoint with a running remote server.

  ```python
  import transformers

  agent = transformers.HfAgent("http://localhost:3000/hf/agent")  # URL that runs the OpenLLM server

  agent.run("Is the following `text` positive or negative?", text="I don't like how this models is generate inputs")
  ```

  Note that only `starcoder` is currently supported for agent feature.

  To use it from the `openllm.client`, do:
  ```python
  import openllm

  client = openllm.client.HTTPClient("http://123.23.21.1:3000")

  client.ask_agent(
      task="Is the following `text` positive or negative?",
      text="What are you thinking about?",
      agent_type="hf",
  )
  ```

  Fixes a Asyncio exception by increasing the timeout
  [#29](https://github.com/bentoml/openllm/issues/29)


## [0.1.6](https://github.com/bentoml/openllm/tree/v0.1.6)

### Changes

- `--quantize` now takes `int8, int4` instead of `8bit, 4bit` to be consistent
  with bitsandbytes concept.

  `openllm CLI` now cached all available model command, allow faster startup time.

  Fixes `openllm start model-id --debug` to filtered out debug message log from
  `bentoml.Server`.

  `--model-id` from `openllm start` now support choice for easier selection.

  Updated `ModelConfig` implementation with **getitem** and auto generation value.

  Cleanup CLI and improve loading time, `openllm start` should be 'blazingly
  fast'.
  [#28](https://github.com/bentoml/openllm/issues/28)


### Features

- Added support for quantization during serving time.

  `openllm start` now support `--quantize int8` and `--quantize int4` `GPTQ`
  quantization support is on the roadmap and currently being worked on.

  `openllm start` now also support `--bettertransformer` to use
  `BetterTransformer` for serving.

  Refactored `openllm.LLMConfig` to be able to use with `__getitem__`:
  `openllm.DollyV2Config()['requirements']`.

  The access order being:
  `__openllm_*__ > self.<key> > __openllm_generation_class__ > __openllm_extras__`.

  Added `towncrier` workflow to easily generate changelog entries

  Added `use_pipeline`, `bettertransformer` flag into ModelSettings

  `LLMConfig` now supported `__dataclass_transform__` protocol to help with
  type-checking

  `openllm download-models` now becomes `openllm download`
  [#27](https://github.com/bentoml/openllm/issues/27)
