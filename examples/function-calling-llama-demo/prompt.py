#@title Set up Functions
from string import Template
from typing import Callable, Dict, Any, List
import json

HVAC_FUNCS =[
{
    "function": "HVAC_CONTROL",
    "description": "Call an API to adjust the AC setting in the car.",
    "arguments": [
        {
            "name": "action",
            "description": """The type of action requested, must be one of the following:
'SET_TEMPERATURE': set, increase, decrease or turn on AC to a desired temperature. Must be used with the temperature argument;
'UP': increase the temperature from current setting. If a specific temperature is given, use SET_TEMPERATURE instead;
'DOWN': decrease the temperature from current setting. If a specific temperature is given, use SET_TEMPERATURE instead;
'ON': turn on the AC;
'OFF': turn off the AC;
            """,
            "enum": ["ON", "OFF", "UP", "DOWN", "SET_TEMPERATURE"],
            "type": "string",
        },
        {
            "name": "temperature",
            "type": "number",
            "description": "Only used together with the type argument is SET_TEMPERATURE",
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
