# cirq-fsim-compiler

**Differentiable and Riemannian synthesis of two-qubit unitaries into FSim gates — free angles or a coupler's calibrated native gate — with a Cirq transformer and correct OpenQASM 3 I/O.**

[![CI](https://github.com/Jaspersands/cirq-fsim-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/Jaspersands/cirq-fsim-compiler/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Cirq](https://img.shields.io/badge/framework-Cirq-teal.svg)](https://quantumai.google/cirq)
[![JAX](https://img.shields.io/badge/autodiff-JAX-red.svg)](https://github.com/google/jax)

**[▶ Interactive demo](web/index.html)** — the browser runs the same synthesiser: pick or randomise a 4×4 target, watch the real loss curve, toggle the fixed native gate, and export the OpenQASM 3 that the actual angles produce.

---

## What it does

A two-qubit target $U_t$ is approximated by

$$U(p) = L_n\,\mathrm{FSim}(\theta_n,\varphi_n)\,L_{n-1}\cdots\mathrm{FSim}(\theta_1,\varphi_1)\,L_0,\qquad L_k = R_k\otimes R'_k,\; R = R_z R_x R_z,$$

minimising the process infidelity $1-|\mathrm{Tr}(U_t^\dagger U)|^2/16$ for $n = 1, 2, 3$ stages and returning the smallest $n$ that meets the tolerance.

* **Free angles** — $(\theta_k,\varphi_k)$ are optimised. Every named two-qubit gate (CNOT, CZ, iSWAP, √iSWAP, SWAP = FSim(π/2, π) up to local Z) needs **one** stage; a Haar-random $U(4)$ needs **two** (two FSims carry four interaction parameters against three KAK coordinates).
* **Calibrated / native** — the angles are frozen to a `CouplerCalibration` (e.g. Sycamore's FSim(π/2, π/6), or √iSWAP) and only the single-qubit dressings are optimised: CZ takes **2** native gates, SWAP and a random $U(4)$ take **3**, iSWAP takes 2 √iSWAPs and cannot be done with one. This is the situation on hardware, and it is what the `SycamoreCalibrationMap` feeds into the transformer per edge.
* **Two solvers, one loss.** `method="euclidean"` runs L-BFGS-B over Euler angles with JAX gradients (NumPy central differences as fallback). `method="riemannian"` keeps the dressings as unitaries and descends on $U(2)^{2(n+1)}\times T^{2n}$ with Cayley retractions $(1-\tfrac{\eta}{2}A)^{-1}(1+\tfrac{\eta}{2}A)V$, so every iterate is exactly unitary. They agree to $10^{-8}$ in fidelity.
* **Circuits, not just gates.** `compile_to_fsim` is a `cirq.transformer`: two-qubit gates are synthesised (native FSims pass through, identical unitaries are cached), gates on ≥3 qubits are expanded with `cirq.decompose` first. Toffoli, Fredkin, CCZ and the 3-qubit QFT compile to ≤2-qubit FSim circuits with fidelity $1-10^{-8}$.
* **OpenQASM 3** — `gate fsim(theta, phi)` is defined with a correct `stdgates.inc` body (`exp(−iθ/2(XX+YY))·cp(−φ)`), an optional `defcal` stub marks it as pulse-native, common one- and two-qubit gates are exported, `LineQubit` indices are preserved, unknown gates raise, and `parse_openqasm3` reads the output back for round-trip checks.

| Module | API |
|---|---|
| `unitary_ansatz` | `fsim_matrix_np`, `single_qubit_zxz_np`, `euler_zxz_from_unitary`, `haar_random_unitary`, `FSimCircuitTemplate(n_stages, fixed_fsim=…)` |
| `riemannian_optimizer` | `DifferentiableFSimSynthesizer(target_infidelity, seed, method)`, `synthesize_unitary_to_fsim`, `DecompositionResult`, `SEED_ANGLES` |
| `riemannian_manifold` | `RiemannianFSimSolver`, `cayley_retract`, `lie_element` |
| `calibration_map` | `CouplerCalibration` (`.sycamore()`, `.sqrt_iswap()`), `SycamoreCalibrationMap` (drift, JSON round-trip) |
| `cirq_transformer` | `compile_to_fsim` (`@cirq.transformer`), `FSimDecomposerTransformer`, `compile_circuit_to_sycamore_fsim` |
| `toffoli_decomposer` | `decompose_{toffoli,fredkin,ccz,qft3}_to_sycamore`, `toffoli_circuit`, `qft_unitary` |
| `openqasm3_io` | `export_to_openqasm3(circuit, include_defcal, expand_fsim)`, `parse_openqasm3` |
| `batch_compiler` | `BatchFSimCompiler(n_workers, calibration)` with stage histogram |
| `cli` | `cirq-fsim synthesize | toffoli | fredkin | ccz | qft3 | qasm | benchmark` |

## Quickstart

```python
import cirq, numpy as np
from cirq_fsim_compiler import (
    synthesize_unitary_to_fsim, CouplerCalibration, SycamoreCalibrationMap,
    compile_to_fsim, decompose_toffoli_to_sycamore, export_to_openqasm3, haar_random_unitary,
)

# Free angles: CNOT is one FSim
r = synthesize_unitary_to_fsim(cirq.unitary(cirq.CNOT), target_infidelity=1e-9)
print(r.n_stages, r.infidelity, r.fsim_angles)          # 1 0.0 [(θ, φ)]

# Calibrated native gate: CZ needs two Sycamore FSim(π/2, π/6)
r = synthesize_unitary_to_fsim(cirq.unitary(cirq.CZ), calibration=CouplerCalibration.sycamore())
print(r.native, r.n_stages, r.infidelity)                # True 2 ~1e-16

# Riemannian solver on a random U(4)
r = synthesize_unitary_to_fsim(haar_random_unitary(4, seed=7), method="riemannian", target_infidelity=1e-9)
print(r.method, r.n_stages, r.infidelity)

# Whole circuits, per-edge calibration, Toffoli, OpenQASM 3
q = cirq.LineQubit.range(3)
circuit = cirq.Circuit(cirq.H(q[0]), cirq.CNOT(q[0], q[1]), cirq.CCNOT(*q))
cal_map = SycamoreCalibrationMap(seed=1)                 # nominal gate + per-edge drift
native = compile_to_fsim(circuit, calibration_map=cal_map)
print(export_to_openqasm3(decompose_toffoli_to_sycamore(*q))[:400])
```

```bash
cirq-fsim synthesize --gate swap --native
cirq-fsim synthesize --gate random --method riemannian --json
cirq-fsim toffoli --qasm-out toffoli.qasm
cirq-fsim benchmark --quick
```

## Honest notes

* "Native Sycamore" in the free-angle mode means *the FSim gate family*; only the calibrated mode restricts to the single gate a real coupler provides.
* The Euclidean solver is faster (tens of iterations vs hundreds); the Riemannian solver exists because the package name promised one, and because it guarantees unitarity of every iterate without a parameterisation chart.
* Depth counts are exact for the tolerance requested; a looser tolerance can lower the count at the cost of fidelity.

## Install & test

```bash
pip install -e ".[dev]"
pytest -v tests/                              # 32 tests
python benchmarks/run_compiler_benchmark.py   # stage counts, solver comparison, 3-qubit gates, drift, batch
```

## Tutorial

[`notebooks/03_sycamore_riemannian_fsim_compiler.ipynb`](notebooks/03_sycamore_riemannian_fsim_compiler.ipynb) — executed outputs included.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0 — Jasper Sands.
