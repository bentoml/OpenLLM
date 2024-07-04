# Developer Guide

This Developer Guide is designed to help you contribute to the OpenLLM project.
Follow these steps to set up your development environment and learn the process
of contributing to our open-source project.

Join our [Discord Channel](https://l.bentoml.com/join-openllm-discord) and reach
out to us if you have any question!

## Table of Contents

- [Developer Guide](#developer-guide)
  - [Table of Contents](#table-of-contents)
  - [Setting Up Your Development Environment](#setting-up-your-development-environment)
  - [Development Workflow](#development-workflow)
    - [Adding new models](#adding-new-models)
    - [Adding bentos](#adding-new-models)
    - [Adding repos](#adding-new-models)

## Setting Up Your Development Environment

Before you can start developing, you'll need to set up your environment:

1. Ensure you have [Git](https://git-scm.com/), and
   [Python3.8+](https://www.python.org/downloads/) installed.
2. Fork the OpenLLM repository from GitHub.
3. Clone the forked repository from GitHub:

   ```bash
   git clone git@github.com:username/OpenLLM.git && cd openllm
   ```

4. Add the OpenLLM upstream remote to your local OpenLLM clone:

   ```bash
   git remote add upstream git@github.com:bentoml/OpenLLM.git
   ```

5. Configure git to pull from the upstream remote:

   ```bash
   git switch main # ensure you're on the main branch
   git fetch upstream --tags
   git branch --set-upstream-to=upstream/main
   ```

## Development Workflow

There are a few ways to contribute to the repository structure for OpenLLM:

### Adding new models

1. [recipe.yaml](./recipe.yaml) contains all related-metadata for generating new LLM-based bentos. To add a new LLM, the following structure should be adhere to:

```yaml
"<model_name>:<model_tag>":
  project: vllm-chat
  service_config:
    name: phi3
    traffic:
      timeout: 300
    resources:
      gpu: 1
      gpu_type: nvidia-tesla-l4
  engine_config:
    model: microsoft/Phi-3-mini-4k-instruct
    max_model_len: 4096
    dtype: half
  chat_template: phi-3
```

- `<model_name>` represents the type of model to be supported. Currently supports `phi3`, `llama2`, `llama3`, `gemma`

- `<model_tag>` emphasizes the type of model and its related metadata. The convention would include `<model_size>-<model_type>-<precision>[-<quantization>]`
  For example:

  - `microsoft/Phi-3-mini-4k-instruct` should be represented as `3.8b-instruct-fp16`.
  - `TheBloke/Llama-2-7B-Chat-AWQ` would be `7b-chat-awq-4bit`

- `project` would be used as the basis for the generated bento. Currently, most models should use `vllm-chat` as default.

- `service_config` entails all BentoML-related [configuration](https://docs.bentoml.com/en/latest/guides/configurations.html) to run this bento.

> [!NOTE]
>
> We recommend to include the following field for `service_config`:
>
> - `name` should be the same as `<model_name>`
> - `resources` includes the available accelerator that can run this models. See more [here](https://docs.bentoml.com/en/latest/guides/configurations.html#resources)

- `engine_config` are fields to be used for vLLM engine. See more supported arguments in [`AsyncEngineArgs`](https://github.com/vllm-project/vllm/blob/7cd2ebb0251fd1fd0eec5c93dac674603a22eddd/vllm/engine/arg_utils.py#L799). We recommend to always include `model`, `max_model_len`, `dtype` and `trust_remote_code`.

- If the model is a chat model, `chat_template` should be used. Add the appropriate `chat_template` under [chat_template directory](./vllm-chat/chat_templates/) should you decide to do so.

2. You can then run `BENTOML_HOME=$(openllm repo default)/bentoml/bentos python make.py <model_name>:<model_tag>` to generate the required bentos.

3. You can then submit a [Pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) to `openllm` with the recipe changes

### Adding bentos

OpenLLM now also manages a [generated bento repository](https://github.com/bentoml/openllm-repo/tree/main). If you update and modify and generated bentos, make sure to update the recipe and added the generated bentos under `bentoml/bentos`.

### Adding repos

If you wish to create a your own managed git repo, you should follow the structure of [bentoml/openllm-repo](https://github.com/bentoml/openllm-repo/tree/main).

To add your custom repo, do `openllm repo add <repo_alias> <git_url>`
