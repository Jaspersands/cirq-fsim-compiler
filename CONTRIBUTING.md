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

## Conventions
- Two-qubit matrices use Cirq's ordering (first qubit = most significant bit); `np.kron(u0, u1)` puts `u0` on qubit 0.
- "Native" means the FSim angles are frozen to a `CouplerCalibration`; "free" means (θ, φ) are optimised.
- Every stochastic routine takes a `seed`.
- Run `pytest -q tests/` and `python benchmarks/run_compiler_benchmark.py --quick` before opening a PR.
