# Contributing to Long-Context Efficiency Benchmarks

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or suggest features
- Check existing issues before creating a new one
- Include detailed information:
  - Description of the issue
  - Steps to reproduce
  - Expected vs actual behavior
  - Environment details (OS, Python version, GPU, etc.)

### Contributing Code

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add tests if applicable
   - Update documentation

4. **Test your changes**
   ```bash
   python examples/test_setup.py
   ```

5. **Commit your changes**
   ```bash
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/long-context-efficiency-kv-experiments.git
cd long-context-efficiency-kv-experiments

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Run tests
python examples/test_setup.py
```

## Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings for public functions and classes
- Keep functions focused and modular

## Areas for Contribution

### High Priority
- Additional attention compression techniques
- Support for more model architectures (Llama, Mistral, etc.)
- Integration with inference frameworks (vLLM, TGI)
- Real-world workload traces

### Medium Priority
- Additional visualization types
- More batching strategies
- Support for distributed inference
- Performance optimizations

### Documentation
- Tutorial notebooks
- Additional examples
- Benchmarking best practices guide
- Case studies

## Testing

- Test your changes thoroughly
- Include unit tests for new functionality
- Verify benchmarks run without errors
- Test on different hardware if possible (CPU/GPU)

## Documentation

- Update README.md if adding new features
- Add docstrings to new functions/classes
- Include usage examples
- Update CHANGELOG.md

## Pull Request Guidelines

- Keep PRs focused on a single feature/fix
- Include a clear description of changes
- Reference related issues
- Ensure all tests pass
- Update documentation as needed

## Questions?

Open an issue for questions or discussion about contributions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
