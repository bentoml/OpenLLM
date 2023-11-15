# Adding a New Model

OpenLLM encourages contributions by welcoming users to incorporate their custom
Large Language Models (LLMs) into the ecosystem. You can set up your development
environment by referring to our
[Developer Guide](https://github.com/bentoml/OpenLLM/blob/main/DEVELOPMENT.md).

## Procedure

All the relevant code for incorporating a new model resides within
[`$GIT_ROOT/openllm-core/src/openllm_core/config`](../openllm-core/src/openllm_core/config/) `model_name` in snake_case.
Here's your roadmap:

- [ ] Generate model configuration file:
      `$GIT_ROOT/openllm-core/src/openllm_core/config/configuration_{model_name}.py`
- [ ] Update `$GIT_ROOT/openllm-core/src/openllm_core/config/__init__.py` to import the new model
- [ ] Add your new model entry in `$GIT_ROOT/openllm-core/src/openllm_core/config/configuration_auto.py` with a tuple of the `model_name` alongside with the `ModelConfig`
- [ ] Run `./tools/update-config-stubs.py`

> [!NOTE]
>
> `$GIT_ROOT` refers to `$(git rev-parse --show-toplevel)`

For a working example, check out any existing model.

### Model Configuration

File Name: `configuration_{model_name}.py`

This file is dedicated to specifying docstrings, default prompt templates,
default parameters, as well as additional fields for the models.

### Entrypoint

After establishing the model config and implementation class, register them in
the `__init__` file, and the tuple under `CONFIG_MAPPING_NAMES` in [openllm-core/src/openllm_core/config/configuration_auto.py#CONFIG_MAPPING_NAMES](https://github.com/bentoml/OpenLLM/blob/main/openllm-core/src/openllm_core/config/configuration_auto.py#L30). Basically you need to register `ModelConfig` classes and
`START_{MODEL}_COMMAND_DOCSTRING` strings.

## Raise a Pull Request

Once you have completed the checklist above, raise a PR and the OpenLLMs
maintainer will review it ASAP. Once the PR is merged, you should be able to see
your model in the next release! 🎉 🎊

### Updating README.md

After a model is added, just ping OpenLLM's maintainers to update the README.md file
with the new model.
