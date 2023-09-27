#@title Set up Functions
from string import Template
from typing import Callable, Dict, Any, List
import json

EXAMPLE_FUNCS =[
{
    "function": "search_bing",
    "description": "Search the web for content on Bing. This allows users to search online/the internet/the web for content.",
    "arguments": [
        {
            "name": "query",
            "type": "string",
            "description": "The search query string"
        }
    ]
},
{
    "function": "search_arxiv",
    "description": "Search for research papers on ArXiv. Make use of AND, OR and NOT operators as appropriate to join terms within the query.",
    "arguments": [
        {
            "name": "query",
            "type": "string",
            "description": "The search query string"
        }
    ]
}]

# Define the roles and markers
B_INST, E_INST = "[INST]", "[/INST]"
B_FUNC, E_FUNC = "<FUNCTIONS>", "</FUNCTIONS>\n\n"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

system_prompt = "you are a helpful AI assitent in a car, that respond to the driver's instructions and helps setting the temperature in the vehicle. You can only communicate using JSON."

def prompt_template(functions: List[Dict[str, Any]]) -> Callable[[str], str]:
    function_list = ""
    for func in functions:
        function_list+=json.dumps(func, indent=4, separators=(',', ': '))
        function_list+="\n\n"
    prompt_base = Template(f"{B_FUNC}{function_list.strip()}{E_FUNC}{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}$user_prompt {E_INST}\n\n")

    def prompt_template(user_prompt: str) -> str:
        return prompt_base.substitute(user_prompt=user_prompt.strip())

    return prompt_template
