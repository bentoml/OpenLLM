from __future__ import annotations
import typing as t

def openllm_to_openai_config(input_dict: dict[str, t.Any], default_config: dict[str, t.Any]) -> dict[str, t.Any]:
    '''Converts OpenLLM config to OpenAI config'''
    input_dict.pop('model', None)  # We don't need the exact model name here
    for key in list(input_dict.keys()):
        if key == 'prompt': continue
        elif key == 'max_tokens':
            default_config['max_new_tokens'] = input_dict[key]
        elif key in default_config:
            default_config[key] = input_dict[key]  # Replace the default config with the input config
        del input_dict[key]  # Remove the key from input_dict
    input_dict['adapter_name'] = None
    input_dict['llm_config'] = default_config
    return input_dict
