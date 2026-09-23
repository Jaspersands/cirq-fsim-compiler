# cirq-fsim-compiler

Decompose two-qubit gates into FSim interactions and compile circuits for tunable-coupler hardware.

[![CI](https://github.com/Jaspersands/cirq-fsim-compiler/actions/workflows/ci.yml/badge.svg)](https://github.com/Jaspersands/cirq-fsim-compiler/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Superconducting processors with tunable couplers implement the fermionic-simulation gate FSim(θ, φ). This package finds decompositions of any two-qubit unitary into FSim gates and single-qubit rotations. The FSim angles can be free, or fixed to the one gate a calibrated coupler performs. As a Cirq transformer it compiles whole circuits, merges runs of gates on the same qubit pair, routes onto a coupling graph, and reads and writes OpenQASM 3.

[Interactive page](https://fsim.jaspersands.com/): the decomposition in the browser, with synthesis in a background worker and OpenQASM export of the fitted angles.

## The decomposition

A target $U_t$ is approximated by

$$U(p) = L_n\,\mathrm{FSim}(\theta_n,\varphi_n)\cdots \mathrm{FSim}(\theta_1,\varphi_1)\,L_0,\qquad L_k = R_k\otimes R'_k,\quad R = R_z R_x R_z,$$

minimising $1 - |\mathrm{Tr}(U_t^\dagger U)|^2/16$ for $n = 1, 2, 3$ and keeping the smallest $n$ that meets the tolerance.

- With free angles, every standard gate (CNOT, CZ, iSWAP, √iSWAP, SWAP) takes one FSim, and a Haar-random unitary takes two.
- With a calibrated coupler only the single-qubit layers are optimised. On Sycamore's FSim(π/2, π/6), CZ takes two gates and SWAP or a random unitary takes three. On √iSWAP, iSWAP takes two.
- There are two solvers for the same loss: L-BFGS-B on Euler angles with JAX gradients, and Riemannian descent that keeps every single-qubit layer exactly unitary through Cayley retractions. They agree to $10^{-8}$.

## Circuits

`compile_to_fsim` is a `cirq.transformer`. Gates on three or more qubits are decomposed first. Each maximal run of operations on one qubit pair, including the single-qubit gates between them, is then merged into a single 4×4 block before synthesis. With a `device_graph` the circuit is first routed with Cirq's `RouteCQC`, and measurements follow their qubits through the routing.

| Calibrated Sycamore gates | per gate | consolidated | routed onto a line |
|---|---|---|---|
| Toffoli | 12 | 10 | 14 |
| Fredkin | 16 | 12 | 16 |
| CCZ | 12 | 10 | 14 |
| 3-qubit QFT | 9 | 9 | 13 |

For reference, a Toffoli needs at least five two-qubit gates of any kind. The consolidated counts are what block-wise synthesis achieves, not a proven optimum for this gate set.

## OpenQASM 3

The exporter defines `gate fsim(theta, phi)` with standard gates, as exp(−iθ/2 (XX + YY)) followed by cp(−φ), so any OpenQASM 3 simulator can run the output. An optional `defcal` stub marks the gate as pulse-native, and measurements go to bit registers.

The importer is a tokenizer and recursive-descent parser for the gate-level subset of the language. It supports:
- several named registers, including OpenQASM 2 `qreg` and `creg`
- the gates in `stdgates.inc`, plus `fsim`
- user gate definitions (nested bodies, gates calling gates)
- register broadcasting, `measure` in both syntaxes, and `reset`
- `defcal` and `cal` blocks, which are skipped
- parameter expressions, which the parser evaluates itself

Anything else, such as control flow, raises `QasmParseError` with a line number. Nothing is passed to `eval`.

## Modules

| Module | Contents |
|---|---|
| `unitary_ansatz` | FSim and Euler-angle matrices, Haar-random unitaries, the decomposition template |
| `riemannian_optimizer` | `DifferentiableFSimSynthesizer` (Euclidean or Riemannian), `synthesize_unitary_to_fsim` |
| `riemannian_manifold` | Cayley retraction and the Riemannian solver |
| `calibration_map` | Coupler calibrations and a per-edge map with drift |
| `cirq_transformer` | `compile_to_fsim`, `FSimDecomposerTransformer` (consolidation, routing, caching) |
| `toffoli_decomposer` | Toffoli, Fredkin, CCZ and 3-qubit QFT circuits and their compiled forms |
| `openqasm3_io` | `export_to_openqasm3`, `parse_openqasm3`, `QasmParseError` |
| `batch_compiler` | `BatchFSimCompiler` with serial, thread or process execution |
| `cli` | `cirq-fsim synthesize | toffoli | fredkin | ccz | qft3 | qasm | benchmark` |

## Examples

```python
import cirq, networkx as nx
from cirq_fsim_compiler import (
    synthesize_unitary_to_fsim, CouplerCalibration, SycamoreCalibrationMap,
    compile_to_fsim, export_to_openqasm3, parse_openqasm3,
)

print(synthesize_unitary_to_fsim(cirq.unitary(cirq.CNOT)).n_stages)                                    # 1
print(synthesize_unitary_to_fsim(cirq.unitary(cirq.CZ), calibration=CouplerCalibration.sycamore()).n_stages)  # 2

q = cirq.LineQubit.range(3)
circuit = cirq.Circuit(cirq.H(q[0]), cirq.CCNOT(*q), cirq.measure(*q, key="m"))
line = nx.Graph([(q[0], q[1]), (q[1], q[2])])
compiled = compile_to_fsim(circuit, calibration_map=SycamoreCalibrationMap(seed=1), device_graph=line)

text = export_to_openqasm3(compiled)
assert parse_openqasm3(text).has_measurements()
```

```bash
cirq-fsim synthesize --gate swap --native
cirq-fsim toffoli --qasm-out toffoli.qasm
cirq-fsim benchmark --quick
```

## Notes

- In free-angle mode "Sycamore" means the FSim family. Only the calibrated mode restricts compilation to the single gate a real coupler provides.
- The Euclidean solver needs fewer iterations. The Riemannian one guarantees that every iterate is unitary without a coordinate chart.
- Batch synthesis is CPU-bound Python, so threads are no faster than serial execution. The process pool gave 1.4× on 480 unitaries on a 10-core laptop, limited by per-worker start-up and JIT compilation. Serial is the default.

## Install

```bash
pip install -e ".[dev]"
pytest tests/                                 # 47 tests
python benchmarks/run_compiler_benchmark.py
```

The notebook [`notebooks/03_sycamore_riemannian_fsim_compiler.ipynb`](notebooks/03_sycamore_riemannian_fsim_compiler.ipynb) has executed outputs. Changes are listed in [CHANGELOG.md](CHANGELOG.md).

Apache-2.0 · Jasper Sands
