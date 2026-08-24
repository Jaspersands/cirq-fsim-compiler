# Native Cirq FSim Compiler via Differentiable Unitary Decomposition

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Cirq Sycamore](https://img.shields.io/badge/Framework-Cirq%20%7C%20Google%20Quantum%20AI-teal.svg)](https://quantumai.google/cirq)
[![JAX Auto-Diff](https://img.shields.io/badge/Autodiff-JAX%20Riemannian-red.svg)](https://github.com/google/jax)

> **JAX-accelerated Riemannian optimization on $U(4)$ Lie manifolds for synthesizing minimum-depth native Google Sycamore $\text{FSim}(\theta, \phi)$ circuits, featuring a drop-in `cirq.Transformer` compiler plugin.**

---

## ⚡ Overview

Google's superconducting quantum processors (Sycamore, Weber) execute native two-parameter fermionic simulation gates:

$$\text{FSim}(\theta, \phi) = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & \cos\theta & -i\sin\theta & 0 \\ 0 & -i\sin\theta & \cos\theta & 0 \\ 0 & 0 & 0 & e^{-i\phi} \end{pmatrix}$$

Standard quantum circuit transpilers decompose algorithms into CNOTs or CZs, which incur high gate counts and error accumulation when re-compiled to Sycamore.

This compiler implements **differentiable unitary synthesis on $U(4)$**:
1. Optimizes parameterized 1-, 2-, and 3-stage $\text{FSim}(\theta, \phi) + 1\text{Q}$ sequences.
2. Directly minimizes the Riemannian Hilbert-Schmidt infidelity loss:
   $$\mathcal{L}(\mathbf{\Theta}) = 1 - \frac{1}{16} \left| \text{Tr}\left( U_{\text{target}}^\dagger U_{\text{ansatz}}(\mathbf{\Theta}) \right) \right|^2$$
3. Integrates directly into Cirq via `FSimDecomposerTransformer` and `compile_circuit_to_sycamore_fsim`.

---

## 📊 Synthesis Depth & Performance

| Target Gate | Standard CNOT Count | Native FSim Stages | Synthesis Infidelity | Compilation Time |
|:---:|:---:|:---:|:---:|:---:|
| **iSWAP** | 2 CNOTs | **1 FSim** | $< 10^{-14}$ | ~15 ms |
| **Givens Rotation** | 2 CNOTs | **1 FSim** | $< 10^{-14}$ | ~18 ms |
| **$\sqrt{\text{iSWAP}}$** | 2 CNOTs | **1 FSim** | $< 10^{-14}$ | ~16 ms |
| **CNOT** | 1 CNOT | **2 FSim** | $< 10^{-12}$ | ~35 ms |
| **CZ** | 1 CZ | **2 FSim** | $< 10^{-12}$ | ~32 ms |
| **SWAP** | 3 CNOTs | **3 FSim** | $< 10^{-12}$ | ~45 ms |
| **Random $U \in U(4)$** | 3 CNOTs | **3 FSim** | $< 10^{-10}$ | ~60 ms |

---

## 🚀 Quickstart

### Installation

```bash
git clone https://github.com/Jaspersands/cirq-fsim-compiler.git
cd cirq-fsim-compiler
pip install -e .
```

### Python API Example

```python
import cirq
from cirq_fsim_compiler import (
    synthesize_unitary_to_fsim,
    compile_circuit_to_sycamore_fsim,
)

# 1. Synthesize a 4x4 matrix into native FSim angles
cnot_mat = cirq.unitary(cirq.CNOT)
result = synthesize_unitary_to_fsim(cnot_mat)

print(f"Stages: {result.n_stages}")
print(f"Infidelity: {result.infidelity:.2e}")
print(f"FSim Angles (theta, phi): {result.fsim_angles}")

# 2. Transpile an arbitrary multi-qubit Cirq Circuit
q = cirq.LineQubit.range(3)
circuit = cirq.Circuit([
    cirq.H(q[0]),
    cirq.CNOT(q[0], q[1]),
    cirq.SWAP(q[1], q[2]),
])

compiled_circuit = compile_circuit_to_sycamore_fsim(circuit)
print("\nNative Sycamore Circuit:")
print(compiled_circuit)
```

---

## 🧪 Testing & Benchmarks

Run unit tests:
```bash
pytest -v tests/
```

Run compilation benchmark:
```bash
python benchmarks/run_compiler_benchmark.py
```

---

## 🌐 Interactive Web Showcase

Open `web/index.html` to experience:
- **In-Browser JAX/WebAssembly Synthesizer**: Input any $4\times 4$ complex matrix or pick presets (CNOT, iSWAP, Fermionic Hop, Random $U(4)$).
- **Real-Time Gradient Optimization**: Watch loss descend live in browser ($10^0 \to 10^{-14}$).
- **Interactive SVG Circuit Diagram**: Displays exact numerical $(\theta_k, \phi_k)$ and Euler angles.
- **Cartan KAK vs Differentiable Compiler Benchmark Metrics**.

---

## 📄 Citation & License

Developed by **Jasper Sands** under the **Apache-2.0 License**.
