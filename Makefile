.PHONY: install clean test jupyter build book serve

install:
	pip install -r requirements.txt

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
	rm -rf _build/

test:
	python -m pytest tests/ -v

jupyter:
	jupyter notebook notebooks/

build:
	jupyter-book build .

book: build
	@echo "Jupyter Book built successfully!"
	@echo "Open _build/html/index.html in your browser"

serve:
	python -m http.server 8000 --directory _build/html

help:
	@echo "Available commands:"
	@echo "  install  - Install dependencies"
	@echo "  clean    - Remove cache files and checkpoints"
	@echo "  test     - Run tests"
	@echo "  jupyter  - Start Jupyter notebook server"
	@echo "  build    - Build Jupyter Book"
	@echo "  book     - Build and show instructions"
	@echo "  serve    - Serve the built book locally"
