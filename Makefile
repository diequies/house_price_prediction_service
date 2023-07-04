.PHONY: tests docs clean

dependencies: 
	@echo "Initializing Git..."
	git init
	@echo "Installing dependencies..."
	poetry install
	poetry run pre-commit install

env: dependencies
	@echo "Activating virtual environment..."
	poetry shell

tests:
	pytest

clean:
	@echo Remove cache files
	rm -rf __pycache__
	rm -rf .pytest_cache

docs:
	@echo Save documentation to docs... 
	pdoc src -o docs --force
	@echo View API documentation... 
	pdoc src --http localhost:8080

lint:
	@echo Linting
	flake8 src
	pylint src
	mypy src