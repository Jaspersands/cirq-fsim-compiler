# Native Cirq FSim Compiler via Differentiable Unitary Decomposition

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Cirq Sycamore](https://img.shields.io/badge/Framework-Cirq%20%7C%20Google%20Quantum%20AI-teal.svg)](https://quantumai.google/cirq)
[![JAX Auto-Diff](https://img.shields.io/badge/Autodiff-JAX%20Riemannian-red.svg)](https://github.com/google/jax)

> **JAX-accelerated Riemannian optimization on $U(4)$ Lie manifolds for synthesizing minimum-depth native Google Sycamore $\text{FSim}(\theta, \phi)$ circuits, featuring hardware calibration drift maps, 3-qubit Toffoli/Fredkin decomposition, and OpenQASM 3.0 export.**

---

## ⚡ Overview & Features (v0.2.0)

- **Differentiable Riemannian Unitary Decomposition**:
  - Minimizes Riemannian Hilbert-Schmidt infidelity on $U(4)$ down to $< 10^{-12}$.
  - Auto-selects 1-, 2-, or 3-stage $\text{FSim}(\theta, \phi) + 1\text{Q}$ Euler gate sequences.
- **Hardware Drift Calibration Mapping (`SycamoreCalibrationMap`)**:
  - Synthesizes circuits specifically on measured, drift-adjusted coupler parameters $(\theta_{\text{cal}}, \phi_{\text{cal}})$ per physical grid edge.
- **3-Qubit Unitary Synthesis (`decompose_toffoli_to_sycamore`, `decompose_fredkin_to_sycamore`)**:
  - Decomposes Toffoli / CCNOT, Fredkin (CSWAP), and 3-qubit QFT into native Sycamore FSim lattices.
- **OpenQASM 3.0 Exporter (`export_to_openqasm3`)**:
  - Direct export with `defcal` parameterized pulse annotations.
- **Batch Multi-Gate Compiler (`BatchFSimCompiler`)**:
  - Vectorized concurrent synthesis for large-scale multi-qubit circuits.

---

## 🚀 Quickstart

```python
import cirq
from cirq_fsim_compiler import (
    synthesize_unitary_to_fsim,
    compile_circuit_to_sycamore_fsim,
    decompose_toffoli_to_sycamore,
    export_to_openqasm3,
    SycamoreCalibrationMap,
)

# 1. Synthesize a 4x4 unitary matrix into native FSim angles
cnot_mat = cirq.unitary(cirq.CNOT)
result = synthesize_unitary_to_fsim(cnot_mat)
print(f"Stages: {result.n_stages}, Infidelity: {result.infidelity:.2e}")

# 2. Decompose 3-Qubit Toffoli Gate
q = cirq.LineQubit.range(3)
toffoli_circuit = decompose_toffoli_to_sycamore(q[0], q[1], q[2])
print(toffoli_circuit)

# 3. Export to OpenQASM 3.0
qasm3 = export_to_openqasm3(toffoli_circuit, file_path="toffoli_sycamore.qasm")
```

---

## 🧪 Testing & Benchmarks

```bash
pytest -v tests/
python benchmarks/run_compiler_benchmark.py
```

---

## 📄 Citation & License

Developed by **Jasper Sands** under the **Apache-2.0 License**.
