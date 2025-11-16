# Contributing to Orchestry

Thank you for your interest in contributing to Orchestry! This guide will help you get started.

## Development Setup

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/Orchestry.git
cd Orchestry

# Create virtual environment and install dependencies
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Set up environment variables
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

## Development Workflow

### Code Quality Tools

We use strict code quality standards:

- **Black** (v24.0+): Code formatting (line length: 100)
- **Ruff** (v0.3+): Fast linting and import sorting
- **Mypy** (v1.9+): Strict type checking
- **Pytest** (v7.4+): Testing framework

### Common Commands

We provide a Makefile for convenience:

```bash
# Format code
make format

# Run linters
make lint

# Type check
make type-check

# Run tests
make test

# Run tests with coverage
make test-cov

# Run all checks (recommended before committing)
make all

# Quick format and lint
make quick

# Clean build artifacts
make clean
```

### Manual Commands

If you prefer not to use Make:

```bash
# Format
black orchestry/ examples/ tests/
ruff check --fix orchestry/ examples/ tests/

# Lint
ruff check orchestry/ examples/ tests/

# Type check
mypy orchestry/

# Test
pytest tests/
pytest --cov=orchestry --cov-report=html  # with coverage
```

## Code Style Guidelines

### Type Hints

All public functions and methods must have complete type annotations:

```python
# Good
def process_data(items: list[str], threshold: float = 0.5) -> dict[str, Any]:
    """Process items and return results."""
    ...

# Bad - missing type hints
def process_data(items, threshold=0.5):
    ...
```

### Docstrings

Use Google-style docstrings for all public functions, classes, and modules:

```python
def calculate_reward(state: State, action: Action) -> float:
    """Calculate the reward for a state-action pair.

    Args:
        state: The current state
        action: The action taken

    Returns:
        The computed reward value

    Raises:
        ValueError: If state or action is invalid
    """
    ...
```

### Modern Python Syntax

Use Python 3.11+ features:

```python
# Use built-in generics (not typing.List, typing.Dict)
def process(items: list[str]) -> dict[str, int]:
    ...

# Use union operator (not Union)
def get_value() -> str | None:
    ...

# Use match statements for complex conditionals
match status:
    case "success":
        return result
    case "error":
        raise error
```

### Import Organization

Imports are automatically sorted by Ruff. The order is:

1. Standard library
2. Third-party packages
3. Local imports

```python
import os
import sys
from pathlib import Path

import numpy as np
from anthropic import Anthropic

from orchestry.marl.trainer import MARLTrainer
from orchestry.tasks.base import TaskConfig
```

## Testing

### Writing Tests

- Place tests in `tests/` directory mirroring package structure
- Use descriptive test names: `test_<feature>_<scenario>`
- Use pytest fixtures for common setup
- Aim for 80%+ code coverage

```python
# tests/marl/test_trajectory.py
import pytest
from orchestry.marl.trajectory import Turn, MultiTurnTrajectory


def test_trajectory_adds_turns_correctly():
    """Test that trajectory correctly adds and tracks turns."""
    traj = MultiTurnTrajectory()
    turn = Turn(agent_id=0, content="Hello", reward=1.0)

    traj.add_turn(turn)

    assert len(traj.turns) == 1
    assert traj.turns[0] == turn


@pytest.mark.slow
def test_full_episode_execution():
    """Integration test for full episode."""
    ...
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/marl/test_trajectory.py

# Run tests matching pattern
pytest -k "test_trajectory"

# Run with coverage
pytest --cov=orchestry --cov-report=html

# Run in parallel (fast)
pytest -n auto
```

## Pre-commit Hooks

Pre-commit hooks run automatically on `git commit`:

- Black formatting
- Ruff linting
- Mypy type checking (on orchestry/ only)
- File cleanup (trailing whitespace, etc.)

To run manually:

```bash
pre-commit run --all-files
```

To skip (not recommended):

```bash
git commit --no-verify
```

## Project Structure

```
orchestry/
├── orchestry/              # Main package
│   ├── __init__.py
│   ├── py.typed           # PEP 561 type marker
│   ├── marl/              # MARL implementation
│   ├── tasks/             # Task definitions
│   ├── legacy/            # Legacy system
│   └── cli/               # CLI entry points
├── tests/                 # Tests (mirrors orchestry/)
├── examples/              # Example scripts
├── configs/               # Configuration files
├── docs/                  # Documentation
├── pyproject.toml         # Project config and dependencies
├── uv.lock                # Locked dependencies
├── Makefile               # Development commands
└── README.md              # Main documentation
```

## Commit Messages

Use clear, descriptive commit messages:

```
feat: add beam search to MARL trainer
fix: resolve type error in value estimator
docs: update installation instructions
test: add trajectory tests
refactor: simplify reward calculation
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `ci`: CI/CD changes

## Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/my-feature`
3. Make your changes
4. Run all checks: `make all`
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

### PR Checklist

- [ ] Code is formatted (Black + Ruff)
- [ ] All lints pass (Ruff)
- [ ] Type checking passes (Mypy)
- [ ] Tests pass and coverage is maintained/improved
- [ ] Documentation updated if needed
- [ ] Changelog entry added (if significant change)
- [ ] Pre-commit hooks pass

## Questions?

- Open an issue for bugs or feature requests
- Check existing documentation in `docs/`
- Read the main README.md

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
