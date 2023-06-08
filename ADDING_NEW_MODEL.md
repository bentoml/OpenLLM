# Adding a New Model

OpenLLM encourages contributions by welcoming users to incorporate their custom
Large Language Models (LLMs) into the ecosystem. You can set up your development
environment by referring to our
[Developer Guide](https://github.com/bentoml/OpenLLM/blob/main/DEVELOPMENT.md).

## Procedure

All the relevant code for incorporating a new model resides within
`src/openllm/models`. Start by creating a new folder named after your
`model_name` in snake_case. Here's your roadmap:

- [ ] Generate model configuration file:
      `src/openllm/models/{model_name}/configuration_{model_name}.py`
- [ ] Establish model implementation files:
      `src/openllm/models/{model_name}/modeling_{runtime}_{model_name}.py`
- [ ] Create module's `__init__.py`:
      `src/openllm/models/{model_name}/__init__.py`
- [ ] Adjust the entrypoints for files at `src/openllm/models/auto/*`
- [ ] Modify the main `__init__.py`: `src/openllm/models/__init__.py`
- [ ] Develop or adjust dummy objects for dependencies, a task exclusive to the
      `utils` directory: `src/openllm/utils/*`

For a working example, check out any pre-implemented model.

> We are developing a CLI command and helper script to generate these files,
> which would further streamline the process. Until then, manual creation is
> necessary.

### Model Configuration

File Name: `configuration_{model_name}.py`

This file is dedicated to specifying docstrings, default prompt templates,
default parameters, as well as additional fields for the models.

### Model Implementation

File Name: `modeling_{runtime}_{model_name}.py`

For each runtime, i.e., torch (default with no prefix), TensorFlow -`tf`, Flax -
`flax`, it is necessary to implement a class that adheres to the `openllm.LLM`
interface. The conventional class name follows the `RuntimeModelName` pattern,
e.g., `FlaxFlanT5`.

### Initialization Files

The `__init__.py` files facilitate intelligent imports, type checking, and
auto-completions for the OpenLLM codebase and CLIs.

### Entrypoint

After establishing the model config and implementation class, register them in
the `auto` folder files. There are four entrypoint files:

- `configuration_auto.py`: Registers `ModelConfig` classes
- `modeling_auto.py`: Registers a model's PyTorch implementation
- `modeling_tf_auto.py`: Registers a model's TensorFlow implementation
- `modeling_flax_auto.py`: Registers a model's Flax implementation

### Dummy Objects

In the `src/openllm/utils` directory, dummy objects are created for each model
and runtime implementation. These specify the dependencies required for each
model.

### Updating README.md

Run `./tools/update-readme.py` to update the README.md file with the new model.

## Raise a Pull Request

Once you have completed the checklist above, raise a PR and the OpenLLMs
maintainer will review it ASAP. Once the PR is merged, you should be able to see
your model in the next release! 🎉 🎊
