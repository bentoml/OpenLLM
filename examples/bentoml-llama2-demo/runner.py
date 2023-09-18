import os
import openllm

#run `openllm models` to find more model IDs of llama2
MODEL_ID = "NousResearch/llama-2-7b-chat-hf"  #@param ["NousResearch/llama-2-7b-chat-hf", "NousResearch/llama-2-13b-chat-hf","NousResearch/llama-2-70b-chat-hf"]
BACKEND = "vllm"  #@param ["pt", "vllm"] # we recommend to use vllm for better performance

os.environ['OPENLLM_MODEL_ID'] = MODEL_ID
os.environ['OPENLLM_BACKEND'] = BACKEND
model = "llama"

llm_config = openllm.AutoConfig.for_model(model)

llm_runner = openllm.Runner(model, llm_config=llm_config)

#download model
llm_runner.download_model()