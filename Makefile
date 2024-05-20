PWD := $(shell pwd)

BENTOML_HOME := $(PWD)/../openllm-repo/bentoml

.PHONY: all
all:
	@rm -rf $(BENTOML_HOME)
	@mkdir -p $(BENTOML_HOME)
	@cd vllm-chat && BENTOML_HOME=$(BENTOML_HOME) CLLAMA_MODEL=llama2:7b-chat bentoml build . --version 7b-chat
	@cd vllm-chat && BENTOML_HOME=$(BENTOML_HOME) CLLAMA_MODEL=llama2:7b bentoml build . --version 7b
	@cd vllm-chat && BENTOML_HOME=$(BENTOML_HOME) CLLAMA_MODEL=mistral:7b-instruct bentoml build . --version 7b-instruct
	@cd vllm-chat && BENTOML_HOME=$(BENTOML_HOME) CLLAMA_MODEL=mistral:7b bentoml build . --version 7b
	@cd vllm-chat && BENTOML_HOME=$(BENTOML_HOME) CLLAMA_MODEL=mixtral:8x7b-instruct bentoml build . --version 8x7b-instruct
	@cd vllm-chat && BENTOML_HOME=$(BENTOML_HOME) CLLAMA_MODEL=mixtral:8x7b bentoml build . --version 8x7b
