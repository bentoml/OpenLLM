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
