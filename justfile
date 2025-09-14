# Python project automation with just

# Format all Python files with ruff
format:
    ruff format .

# Lint with ruff
lint:
    ruff check --fix .

# Run the type checker
type:
    ty check

# Run both format, lint, and type check
all: format lint type

test:
    uv run pytest

help:
    @just --list
