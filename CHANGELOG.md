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

## [0.3.5](https://github.com/bentoml/openllm/tree/v0.3.5)

### Features

- Added support for continuous batching via vLLM

  Currently benchmark shows that 100 concurrent requests shows around 1218 TPS on 1 A100 running meta-llama/Llama-2-13b-chat-hf
  [#349](https://github.com/bentoml/openllm/issues/349)


### Bug fix

- Set a default serialisation for all models.

  Currently, only Llama 2 will use safetensors as default format. For all other models, if they have safetensors format, then it will can be opt-int via `--serialisation safetensors`
  [#355](https://github.com/bentoml/openllm/issues/355)


## [0.3.4](https://github.com/bentoml/openllm/tree/v0.3.4)

### Bug fix

- vLLM now should support safetensors loading format, so `--serlisation` should be agnostic of backend now

  Removed some legacy check and default behaviour
  [#324](https://github.com/bentoml/openllm/issues/324)


## [0.3.3](https://github.com/bentoml/openllm/tree/v0.3.3)
No significant changes.


## [0.3.2](https://github.com/bentoml/openllm/tree/v0.3.2)
No significant changes.


## [0.3.1](https://github.com/bentoml/openllm/tree/v0.3.1)

### Changes

- revert back to only release pure wheels

  disable compiling wheels for now once we move to different implementation
  [#304](https://github.com/bentoml/openllm/issues/304)


## [0.3.0](https://github.com/bentoml/openllm/tree/v0.3.0)

### Backwards-incompatible Changes

- All environment variable now will be more simplified, without the need for the specific model prefix

  For example: OPENLLM_LLAMA_GENERATION_MAX_NEW_TOKENS now becomes OPENLLM_GENERATION_MAX_NEW_TOKENS

  Unify some misc environment variable. To switch different backend, one can use `--backend` for both `start` and `build`

  ```bash
  openllm start llama --backend vllm
  ```

  or the environment variable `OPENLLM_BACKEND`

  ```bash
  OPENLLM_BACKEND=vllm openllm start llama
  ```

  `openllm.Runner` now will default to try download the model the first time if the model is not available, and get the cached in model store consequently

  Model serialisation now updated to a new API version with more clear name change, kindly ask users to do `openllm prune -y --include-bentos` and update to
  this current version of openllm
  [#283](https://github.com/bentoml/openllm/issues/283)


### Refactor

- Refactor GPTQ to use official implementation from transformers>=4.32
  [#297](https://github.com/bentoml/openllm/issues/297)


### Features

- Added support for vLLM streaming

  This can now be accessed via `/v1/generate_stream`
  [#260](https://github.com/bentoml/openllm/issues/260)


## [0.2.27](https://github.com/bentoml/openllm/tree/v0.2.27)

### Changes

- Define specific style guideline for the project. See
  [STYLE.md](https://github.com/bentoml/OpenLLM/blob/main/STYLE.md) for more
  information.
  [#168](https://github.com/bentoml/openllm/issues/168)


### Refactor

- Expose all extension via `openllm extension`

  Added a separate section for all extension with the CLI. `openllm playground` is now considered as an extension

  introduce compiled wheels gradually

  added a easy `cz.py` for code golf and LOC
  [#191](https://github.com/bentoml/openllm/issues/191)
- Refactor openllm_js to openllm-node for initial node library development
  [#199](https://github.com/bentoml/openllm/issues/199)
- OpenLLM now comprise of three packages:

  - `openllm-core`: main building blocks of OpenLLM, that doesn't depend on transformers and heavy DL libraries
  - `openllm-client`: The implementation of `openllm.client`
  - `openllm`: = `openllm-core` + `openllm-client` + DL features (under `openllm-python`)

  OpenLLM now will provide `start-grpc` as opt-in. If you wan to use `openllm start-grpc`, make sure to install
  with `pip install "openllm[grpc]"`
  [#249](https://github.com/bentoml/openllm/issues/249)


### Features

- OpenLLM now provides SSE support

  > [!NOTE]
  > For this to work, you must install BentoML>=1.1.2:
  > `pip install -U bentoml>=1.1.2`

  The endpoint can be accessed via `/v1/generate_stream`

  > [!NOTE]
  > Curl does in fact does support SSE by passing in `-N`
  [#240](https://github.com/bentoml/openllm/issues/240)


## [0.2.26](https://github.com/bentoml/openllm/tree/v0.2.26)

### Features

- Added a generic embedding implementation largely based on https://github.com/bentoml/sentence-embedding-bento
  For all unsupported models.
  [#227](https://github.com/bentoml/openllm/issues/227)


### Bug fix

- Fixes correct directory for building standalone installer
  [#228](https://github.com/bentoml/openllm/issues/228)


## [0.2.25](https://github.com/bentoml/openllm/tree/v0.2.25)

### Features

- OpenLLM now include a community-maintained ClojureScript UI, Thanks @GutZuFusss

  See [this README.md](/openllm-contrib/clojure/README.md) for more information

  OpenLLM will also include a `--cors` to enable start with cors enabled.
  [#89](https://github.com/bentoml/openllm/issues/89)
- Nightly wheels now can be installed via `test.pypi.org`:

  ```bash
  pip install -i https://test.pypi.org/simple/ openllm
  ```
  [#215](https://github.com/bentoml/openllm/issues/215)
- Running vLLM with Falcon is now supported
  [#223](https://github.com/bentoml/openllm/issues/223)


## [0.2.24](https://github.com/bentoml/openllm/tree/v0.2.24)
No significant changes.


## [0.2.23](https://github.com/bentoml/openllm/tree/v0.2.23)

### Features

- Added all compiled wheels for all supported Python version for Linux and MacOS
  [#201](https://github.com/bentoml/openllm/issues/201)


## [0.2.22](https://github.com/bentoml/openllm/tree/v0.2.22)
No significant changes.


## [0.2.21](https://github.com/bentoml/openllm/tree/v0.2.21)

### Changes

- Added lazy eval for compiled modules, which should speed up overall import time
  [#200](https://github.com/bentoml/openllm/issues/200)


### Bug fix

- Fixes compiled wheels ignoring client libraries
  [#197](https://github.com/bentoml/openllm/issues/197)


## [0.2.20](https://github.com/bentoml/openllm/tree/v0.2.20)
No significant changes.


## [0.2.19](https://github.com/bentoml/openllm/tree/v0.2.19)
No significant changes.


## [0.2.18](https://github.com/bentoml/openllm/tree/v0.2.18)

### Changes

- Runners server now will always spawn one instance regardless of the configuration of workers-per-resource

  i.e: If CUDA_VISIBLE_DEVICES=0,1,2 and `--workers-per-resource=0.5`, then runners will only use `0,1` index
  [#189](https://github.com/bentoml/openllm/issues/189)


### Features

- OpenLLM now can also be installed via brew tap:
  ```bash
  brew tap bentoml/openllm https://github.com/bentoml/openllm

  brew install openllm
  ```
  [#190](https://github.com/bentoml/openllm/issues/190)


## [0.2.17](https://github.com/bentoml/openllm/tree/v0.2.17)

### Changes

- Updated loading logics for PyTorch and vLLM where it will check for initialized parameters after placing to correct devices

  Added xformers to base container for requirements on vLLM-based container
  [#185](https://github.com/bentoml/openllm/issues/185)


### Features

- Importing models now won't load into memory if it is a remote ID. Note that for GPTQ and local model the behaviour is unchanged.

  Fixes that when there is one GPU, we ensure to call `to('cuda')` to place the model onto the memory. Note that the GPU must have
  enough VRAM to offload this model onto the GPU.
  [#183](https://github.com/bentoml/openllm/issues/183)


## [0.2.16](https://github.com/bentoml/openllm/tree/v0.2.16)
No significant changes.


## [0.2.15](https://github.com/bentoml/openllm/tree/v0.2.15)
No significant changes.


## [0.2.14](https://github.com/bentoml/openllm/tree/v0.2.14)

### Bug fix

- Fixes a bug with `EnvVarMixin` where it didn't respect environment variable for specific fields

  This inherently provide a confusing behaviour with `--model-id`. This is now has been addressed with main

  The base docker will now also include a installation of xformers from source, locked at a given hash, since the latest release of xformers
  are too old and would fail with vLLM when running within the k8s
  [#181](https://github.com/bentoml/openllm/issues/181)


## [0.2.13](https://github.com/bentoml/openllm/tree/v0.2.13)
No significant changes.


## [0.2.12](https://github.com/bentoml/openllm/tree/v0.2.12)

### Features

- Added support for base container with OpenLLM. The base container will contains all necessary requirements
  to run OpenLLM. Currently it does included compiled version of FlashAttention v2, vLLM, AutoGPTQ and triton.

  This will now be the base image for all future BentoLLM. The image will also be published to public GHCR.

  To extend and use this image into your bento, simply specify ``base_image`` under ``bentofile.yaml``:

  ```yaml
  docker:
    base_image: ghcr.io/bentoml/openllm:<hash>
  ```

  The release strategy would include:
  - versioning of ``ghcr.io/bentoml/openllm:sha-<sha1>`` for every commit to main, ``ghcr.io/bentoml/openllm:0.2.11`` for specific release version
  - alias ``latest`` will be managed with docker/build-push-action (discouraged)

  Note that all these images include compiled kernels that has been tested on Ampere GPUs with CUDA 11.8.

  To quickly run the image, do the following:

  ```bash
  docker run --rm --gpus all -it -v /home/ubuntu/.local/share/bentoml:/tmp/bentoml -e BENTOML_HOME=/tmp/bentoml \
              -e OPENLLM_USE_LOCAL_LATEST=True -e OPENLLM_BACKEND=vllm ghcr.io/bentoml/openllm:2b5e96f90ad314f54e07b5b31e386e7d688d9bb2 start llama --model-id meta-llama/Llama-2-7b-chat-hf --workers-per-resource conserved --debug`
  ```

  In conjunction with this, OpenLLM now also have a set of small CLI utilities via ``openllm ext`` for ease-of-use

  General fixes around codebase bytecode optimization

  Fixes logs output to filter correct level based on ``--debug`` and ``--quiet``

  ``openllm build`` now will default run model check locally. To skip it pass in ``--fast`` (before this is the default behaviour, but ``--no-fast`` as default makes more sense here as ``openllm build`` should also be able to run standalone)

  All ``LlaMA`` namespace has been renamed to ``Llama`` (internal change and shouldn't affect end users)

  ``openllm.AutoModel.for_model`` now will always return the instance. Runner kwargs will be handled via create_runner
  [#142](https://github.com/bentoml/openllm/issues/142)
- All OpenLLM base container now are scanned for security vulnerabilities using
  trivy (both SBOM mode and CVE)
  [#169](https://github.com/bentoml/openllm/issues/169)


## [0.2.11](https://github.com/bentoml/openllm/tree/v0.2.11)

### Features

- Added embeddings support for T5 and ChatGLM
  [#153](https://github.com/bentoml/openllm/issues/153)


## [0.2.10](https://github.com/bentoml/openllm/tree/v0.2.10)

### Features

- Added installing with git-archival support

  ```bash
  pip install "https://github.com/bentoml/openllm/archive/main.tar.gz"
  ```
  [#143](https://github.com/bentoml/openllm/issues/143)
- Users now can call ``client.embed`` to get the embeddings from the running LLMServer

      ```python
      client = openllm.client.HTTPClient("http://localhost:3000")

      client.embed("Hello World")
      client.embed(["Hello", "World"])
      ```

  > **Note:** The ``client.embed`` is currently only implemnted for ``openllm.client.HTTPClient`` and ``openllm.client.AsyncHTTPClient``

  Users can also query embeddings directly from the CLI, via ``openllm embed``:

      ```bash
      $ openllm embed --endpoint localhost:3000 "Hello World" "My name is Susan"

      [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
      ```
  [#146](https://github.com/bentoml/openllm/issues/146)


### Bug fix

- Fixes model location while running within BentoContainer correctly

  This makes sure that the tags and model path are inferred correctly, based on BENTO_PATH and /.dockerenv
  [#141](https://github.com/bentoml/openllm/issues/141)


## [0.2.9](https://github.com/bentoml/openllm/tree/v0.2.9)
No significant changes.


## [0.2.8](https://github.com/bentoml/openllm/tree/v0.2.8)

### Features

- APIs for LLMService are now provisional based on the capabilities of the LLM.

  The following APIs are considered provisional:

  - `/v1/embeddings`: This will be available if the LLM supports embeddings (i.e: ``LLM.embeddings`` is implemented. Example model are ``llama``)
  - `/hf/agent`: This will be available if LLM supports running HF agents (i.e: ``LLM.generate_one`` is implemented. Example model are ``starcoder``, ``falcon``.)
  - `POST /v1/adapters` and `GET /v1/adapters`: This will be available if the server is running with LoRA weights

  ``openllm.LLMRunner`` now include three additional boolean:
  - `runner.supports_embeddings`: Whether this runner supports embeddings
  - `runner.supports_hf_agent`: Whether this runner support HF agents
  - `runner.has_adapters`: Whether this runner is loaded with LoRA adapters.

  Optimized ``openllm.models``'s bytecode performance
  [#133](https://github.com/bentoml/openllm/issues/133)


## [0.2.7](https://github.com/bentoml/openllm/tree/v0.2.7)
No significant changes.


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

  Refactored `openllm.LLMConfig` to be able to use with `__getitem__`:
  `openllm.DollyV2Config()['requirements']`.

  The access order being:
  `__openllm_*__ > self.<key> > __openllm_generation_class__ > __openllm_extras__`.

  Added `towncrier` workflow to easily generate changelog entries

  `LLMConfig` now supported `__dataclass_transform__` protocol to help with
  type-checking

  `openllm download-models` now becomes `openllm download`
  [#27](https://github.com/bentoml/openllm/issues/27)
