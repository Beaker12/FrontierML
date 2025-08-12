.PHONY: help setup clean jupyter test lint format install-dev

help:
	@echo "Available commands:"
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  clean      - Remove virtual environment and cache files"
	@echo "  jupyter    - Start Jupyter Lab"
	@echo "  test       - Run tests"
	@echo "  lint       - Run linting"
	@echo "  format     - Format code with black"
	@echo "  install-dev - Install development dependencies"

setup:
	python -m venv venv
	./venv/Scripts/pip install --upgrade pip
	./venv/Scripts/pip install -r requirements.txt
	@echo "Virtual environment created. Activate with: venv\\Scripts\\activate"

clean:
	rm -rf venv/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

jupyter:
	jupyter lab --notebook-dir=notebooks

test:
	python -m pytest tests/ -v --cov=src/

lint:
	flake8 src/ tests/
	mypy src/

format:
	black src/ tests/
	black notebooks/*.ipynb --target-version py38

install-dev:
	pip install -e .
