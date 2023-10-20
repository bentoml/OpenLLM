# Adding a New Model

OpenLLM encourages contributions by welcoming users to incorporate their custom
Large Language Models (LLMs) into the ecosystem. You can set up your development
environment by referring to our
[Developer Guide](https://github.com/bentoml/OpenLLM/blob/main/DEVELOPMENT.md).

## Procedure

All the relevant code for incorporating a new model resides within
[`$GIT_ROOT/openllm-python/src/openllm/models`](./src/openllm/models/) `model_name` in snake_case.
Here's your roadmap:

- [ ] Generate model configuration file:
      `$GIT_ROOT/openllm-core/src/openllm_core/config/configuration_{model_name}.py`
- [ ] Establish model implementation files:
      `$GIT_ROOT/openllm-python/src/openllm/models/{model_name}/modeling_{runtime}_{model_name}.py`
- [ ] Create module's `__init__.py`:
      `$GIT_ROOT/openllm-python/src/openllm/models/{model_name}/__init__.py`
- [ ] Adjust the entrypoints for files at `$GIT_ROOT/openllm-python/src/openllm/models/auto/*` If it is a
      new runtime, then add it a `$GIT_ROOT/openllm-python/src/openllm/models/auto/modeling_{runtime}_auto.py`.
      See the other auto runtime for example.
- [ ] Run the following script: `$GIT_ROOT/tools/update-models-import.py`
- [ ] Run the following to update stubs: `hatch run check-stubs`

> [!NOTE]
>
> `$GIT_ROOT` refers to `$(git rev-parse --show-toplevel)`

For a working example, check out any existing model.

### Model Configuration

File Name: `configuration_{model_name}.py`

This file is dedicated to specifying docstrings, default prompt templates,
default parameters, as well as additional fields for the models.

### Model Implementation

File Name: `modeling_{runtime}_{model_name}.py`

For each runtime, i.e., torch (default with no prefix), vLLM - `vllm`, it is necessary to implement a class that adheres to the `openllm.LLM`
interface. The conventional class name follows the `RuntimeModelName` pattern,
e.g., `VLLMFlanT5`.

### Initialization Files

The `__init__.py` files facilitate intelligent imports, type checking, and
auto-completions for the OpenLLM codebase and CLIs.

### Entrypoint

After establishing the model config and implementation class, register them in
the `auto` folder files. There are four entrypoint files:

- `configuration_auto.py`: Registers `ModelConfig` classes
- `modeling_auto.py`: Registers a model's PyTorch implementation
- `modeling_vllm_auto.py`: Registers a model's vLLM implementation

### Updating README.md

Run `./tools/update-readme.py` to update the README.md file with the new model.

## Raise a Pull Request

Once you have completed the checklist above, raise a PR and the OpenLLMs
maintainer will review it ASAP. Once the PR is merged, you should be able to see
your model in the next release! 🎉 🎊
