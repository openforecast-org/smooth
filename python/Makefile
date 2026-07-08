.PHONY: all lint mypy test install environment

SRC_DIR = smooth

lint:
	ruff check src/$(SRC_DIR)
	ruff format --check src/$(SRC_DIR)

mypy:
	mypy src/$(SRC_DIR)

test:
	pytest tests/

install:
	pip install -e ".[dev]"

environment:
	(\
		echo "> Creating venv"; \
		python -m venv .venv; \
		source .venv/bin/activate; \
		echo "> Installing local package in editable mode"; \
		pip install -e ".[dev]"; \
		echo "> Making venv available in jupyter notebooks"; \
		python -m ipykernel install --name=$(SRC_DIR); \
		jupyter kernelspec list; \
		echo "> Installing pre-commit"; \
		pre-commit install; \
	)

clean:
	echo "> Removing virtual environment"
	rm -r .venv
	echo "> Uninstalling from jupyter"
	jupyter kernelspec uninstall $(SRC_DIR)
