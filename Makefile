SHELL=/bin/bash

ALL: help

# print only the ones fullfilling the regex pattern.
help:
	@echo "Available targets:"
	@grep -E '^[a-z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@grep -E '^[a-z0-9_-]+:' $(MAKEFILE_LIST) | grep -v '##' | awk 'BEGIN {FS = ":"}; {printf "  \033[36m%-20s\033[0m\n", $$1}'


compile-requirements:
	uv pip compile infrastructure/requirements.in --output-file infrastructure/requirements.txt

create-local-env:
	python -m venv .venv

install-requirements:
	.venv/bin/pip install -r infrastructure/requirements.txt

install-dev-requirements:
	.venv/bin/pip install -r infrastructure/requirements-dev.txt


setup: create-local-env install-requirements
	@echo "To activate the virtual environment, run:"
	@echo "  source .venv/bin/activate"

setup-dev: setup install-dev-requirements

run-pc:
	.venv/bin/pre-commit run --all-files
