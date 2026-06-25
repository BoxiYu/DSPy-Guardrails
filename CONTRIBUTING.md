# Contributing to dspy-guardrails

Thank you for your interest in contributing to dspy-guardrails! This document provides guidelines and instructions for contributing.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Commit Messages](#commit-messages)

---

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all experience levels.

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what is best for the project
- Show empathy towards other contributors

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment tool (venv, uv, or similar)

### Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/dspy-guardrails.git
cd dspy-guardrails
```

---

## Development Setup

### 1. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Using uv (recommended)
uv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Development installation with all dependencies
pip install -e ".[dev,all]"

# Or minimal installation
pip install -e ".[dev]"
```

### 3. Verify Installation

```bash
# Run tests to verify setup
python tests/test_guardrails.py

# Check imports
python -c "from dspy_guardrails import guardrail; print('Setup OK')"
```

---

## How to Contribute

### Reporting Bugs

1. **Search existing issues** to avoid duplicates
2. **Create a new issue** with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Python version and OS
   - Minimal code example

**Bug Report Template:**

```markdown
## Description
[Clear description of the bug]

## Steps to Reproduce
1. ...
2. ...
3. ...

## Expected Behavior
[What should happen]

## Actual Behavior
[What actually happens]

## Environment
- Python version:
- OS:
- dspy-guardrails version:

## Code Example
```python
# Minimal code to reproduce
```
```

### Suggesting Features

1. **Check roadmap/issues** for existing proposals
2. **Create a feature request** with:
   - Use case description
   - Proposed solution
   - Alternative approaches considered
   - Impact on existing functionality

### Contributing Code

1. **Find or create an issue** for the work
2. **Comment on the issue** to claim it
3. **Fork and create a branch** from `main`
4. **Implement changes** following our guidelines
5. **Write tests** for new functionality
6. **Submit a pull request**

---

## Code Style

### Python Style

We follow PEP 8 with some modifications:

```bash
# Format code with Black
black src/ tests/

# Check with Ruff
ruff check src/ tests/

# Type checking with mypy
mypy src/
```

### Style Guidelines

| Rule | Example |
|------|---------|
| Line length | 88 characters (Black default) |
| Imports | Sorted, grouped (stdlib, third-party, local) |
| Docstrings | Google style |
| Type hints | Required for public APIs |
| Variable names | snake_case |
| Class names | PascalCase |
| Constants | UPPER_SNAKE_CASE |

### Docstring Format

```python
def detect_injection(text: str, threshold: float = 0.5) -> bool:
    """Detect prompt injection attacks in text.

    Args:
        text: The input text to analyze.
        threshold: Detection threshold (0.0-1.0). Default is 0.5.

    Returns:
        True if no injection detected, False otherwise.

    Raises:
        ValueError: If threshold is not between 0.0 and 1.0.

    Example:
        >>> detect_injection("Hello world")
        True
        >>> detect_injection("Ignore all instructions")
        False
    """
```

### Import Order

```python
# Standard library
import os
import re
from typing import Optional, List

# Third-party
import dspy
from pydantic import BaseModel

# Local
from dspy_guardrails import guardrail
from dspy_guardrails.constraints import Constraint
```

---

## Testing

### Running Tests

```bash
# Run all tests
python tests/test_guardrails.py

# Run with pytest (more options)
pytest tests/test_guardrails.py -v

# Run specific test class
pytest tests/test_guardrails.py::TestPromptInjection -v

# Run with coverage
pytest tests/ --cov=src/dspy_guardrails --cov-report=term-missing
```

### Writing Tests

```python
import unittest
from dspy_guardrails import guardrail

class TestNewFeature(unittest.TestCase):
    """Tests for the new feature."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_cases = [
            ("safe input", True),
            ("unsafe input", False),
        ]

    def test_basic_functionality(self):
        """Test basic feature behavior."""
        result = guardrail.new_feature("test")
        self.assertIsNotNone(result)

    def test_edge_cases(self):
        """Test edge cases."""
        # Empty input
        self.assertTrue(guardrail.new_feature(""))

        # Unicode input
        self.assertTrue(guardrail.new_feature("你好世界"))

    def test_expected_failures(self):
        """Test that known attacks are detected."""
        for text, expected in self.test_cases:
            with self.subTest(text=text):
                result = guardrail.new_feature(text)
                self.assertEqual(result, expected)
```

### Test Requirements

- [ ] All new code must have tests
- [ ] Tests must pass before PR can be merged
- [ ] Maintain or improve code coverage
- [ ] Include both positive and negative test cases
- [ ] Test edge cases (empty strings, unicode, long inputs)

---

## Pull Request Process

### Before Submitting

1. **Update from main**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**
   ```bash
   # Format
   black src/ tests/

   # Lint
   ruff check src/ tests/

   # Type check
   mypy src/

   # Tests
   python tests/test_guardrails.py
   ```

3. **Update documentation** if needed

### PR Template

```markdown
## Description
[What does this PR do?]

## Related Issue
Fixes #[issue number]

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainer
3. **Address feedback** if requested
4. **Merge** after approval

---

## Commit Messages

### Format

```
type(scope): short description

[optional body]

[optional footer]
```

### Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change, no feature/fix |
| `test` | Adding/updating tests |
| `chore` | Maintenance tasks |

### Examples

```bash
# Feature
feat(injection): add Chinese pattern detection

# Bug fix
fix(pii): correct phone regex for international formats

# Documentation
docs(readme): update installation instructions

# Refactoring
refactor(guardrail): simplify score calculation logic
```

### Good Commit Messages

```
feat(mcp): add reverse shell detection

- Add patterns for common reverse shell commands
- Support bash, nc, python, and perl variants
- Include test cases for each pattern type

Closes #42
```

### Bad Commit Messages

```
# Too vague
fix bug

# No type
added new feature

# Too long first line
feat: implemented a new really long feature that does many things including detection and filtering and also some other stuff
```

---

## Project Structure

When adding new files, follow this structure:

```
src/dspy_guardrails/
├── __init__.py          # Public exports
├── guardrail.py         # Core detection functions
├── llm_guardrail.py     # LLM-based detection
├── decorators.py        # @Guarded decorator
├── constraints.py       # Constraint system
├── module.py            # GuardedModule base
├── mcp/                 # MCP security module
│   ├── __init__.py
│   └── ...
└── redteam/             # Red team tools
    ├── __init__.py
    └── ...
```

---

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: Search existing issues or create new one
- **Discussions**: Use GitHub Discussions for questions

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to dspy-guardrails!
