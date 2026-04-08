# Contributing to numpy2

Thank you for your interest in contributing to **numpy2**! We welcome all contributions, from bug reports to feature requests to code changes.

## Getting Started

### 1. Fork and Clone

```bash
git clone https://github.com/maheshmakvana/numpy2.git
cd numpy2
```

### 2. Create Development Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -e ".[dev]"
```

### 3. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

## Development Guidelines

### Code Style

- Follow PEP 8 guidelines
- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use type hints where appropriate

### Testing

All new features must include tests:

```bash
pytest tests/ -v
pytest tests/ --cov=numpy2  # Check coverage
```

### Documentation

- Update README.md for user-facing features
- Add docstrings to all functions
- Include examples in docstrings

### Commit Messages

Write clear, descriptive commit messages:

```
[TYPE] Brief description

Longer explanation of the change and why it was made.

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Reporting Issues

When reporting bugs, include:

1. Python version
2. numpy2 version
3. Minimal code example that reproduces the bug
4. Expected vs. actual behavior
5. Error traceback (if applicable)

## Feature Requests

When suggesting features:

1. Explain the use case
2. Provide example code
3. Explain how it solves a problem
4. Consider potential edge cases

## Pull Request Process

1. Update tests and documentation
2. Ensure all tests pass: `pytest tests/ -v`
3. Check code style: `flake8 numpy2/`
4. Run type checker: `mypy numpy2/`
5. Create pull request with clear description

## Code Review Process

- Maintainers will review your PR
- May ask for changes or clarifications
- Once approved, we'll merge your contribution

## Questions?

Feel free to:
- Open an issue on GitHub
- Start a discussion
- Contact: mahesh.makvana@example.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to numpy2! 🙏**
