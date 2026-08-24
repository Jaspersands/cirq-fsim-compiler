# Contributing to cirq-fsim-compiler

Thank you for contributing to **cirq-fsim-compiler**!

## Development Setup

```bash
git clone https://github.com/Jaspersands/cirq-fsim-compiler.git
cd cirq-fsim-compiler
pip install -e ".[dev]"
pytest -v tests/
```

## Pull Request Guidelines
- Verify synthesis convergence on $U(4)$ with `< 1e-4` target infidelity.
- Ensure Cirq transformer compatibility.
