import openllm
llm_runner = openllm.Runner("llama",  model_id="Trelis/Llama-2-7b-chat-hf-function-calling-v2", backend='vllm', serialisation='legacy', temperature=0.2)

#download model
llm_runner.download_model()