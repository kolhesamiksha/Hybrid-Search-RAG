# Variables
DIST_NAME := hybrid-rag  # Poetry package name
PACKAGE_NAME := hybrid_rag  # Python package/module name
SRC_DIR := $(PACKAGE_NAME)/src  # Source directory
TEST_DIR := tests  # Test directory
.PRECOMMIT := .pre-commit-config.yaml

.PHONY: help install test build clean lint format type-check codespell install-precommit run-precommit

# Display available commands
help:
	@echo "Available commands:"
	@echo "  make install       - Install the package in editable mode with dev dependencies"
	@echo "  make test          - Run tests with pytest"
	@echo "  make build         - Build the package as a wheel file"
	@echo "  make clean         - Clean up build artifacts"
	@echo "  make lint          - Run linting with ruff and flake8"
	@echo "  make format        - Format code with black"
	@echo "  make type-check    - Run type checking with mypy"
	@echo "  make codespell     - Check for typos using codespell"

# Install dependencies (editable mode for development)
install:
	@echo "Installing dependencies with Poetry..."
	poetry install --with lint,dev,typing,codespell

# Install pre-comit dependencies
install-precommit:
	pip install pre-commit
	pre-commit install

# Run all pre-commits
run-precommit:
	pre-commit run --all-files

# Run tests
test:
	@echo "Running tests with pytest..."
	poetry run pytest $(TEST_DIR)

# Build the package
build:
	@echo "Building the package as a wheel file..."
	poetry build

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf dist build *.egg-info .mypy_cache .ruff_cache .pytest_cache

#Optional below: IF .pre-commit-config.yaml file is not Present
# Lint code
lint:
	@echo "Running linting with ruff and flake8..."
	poetry run ruff $(SRC_DIR)
	poetry run flake8 $(SRC_DIR)

# Format code
format:
	@echo "Formatting code with black..."
	poetry run black $(SRC_DIR) $(TEST_DIR)

# Type-checking
type-check:
	@echo "Running type checks with mypy..."
	poetry run mypy $(SRC_DIR)

# Check for typos
codespell:
	@echo "Checking for typos with codespell..."
	poetry run codespell $(SRC_DIR) $(TEST_DIR)
