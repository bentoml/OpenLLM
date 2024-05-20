PWD := $(shell pwd)

BENTOML_HOME := $(PWD)/../openllm-repo/bentoml

.PHONY: all
all:
	@rm -rf $(BENTOML_HOME)
	@mkdir -p $(BENTOML_HOME)
	@BENTOML_HOME=$(BENTOML_HOME) python make.py
