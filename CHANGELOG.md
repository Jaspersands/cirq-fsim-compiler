# Changelog

## 0.4.0 — 2026-09-23

Changes from an adversarial review of 0.3.

### Fixed
- The OpenQASM 3 importer called `eval()` on parameter expressions, stripped gate bodies with a regex that broke on nested braces, and required the register to be named `q`. It is now a tokenizer and recursive-descent parser with its own expression evaluator, named registers (and `qreg`/`creg`), user gate definitions inlined with parameters, broadcasting, both measurement syntaxes and `reset`. Errors raise `QasmParseError` with a line number.
- Export no longer refuses measurements; they are written to bit registers.

### Added
- Two-qubit block consolidation in the transformer (default on): Toffoli 12 → 10, Fredkin 16 → 12 calibrated Sycamore gates.
- Routing onto a coupling graph (`device_graph`, Cirq `RouteCQC`); measurements follow their qubits.
- `BatchFSimCompiler(executor=...)`: spawn-based process pool with single-threaded workers, per-thread synthesisers, deterministic per-item seeds.

### Changed
- Dependencies: PennyLane removed (never imported); `cirq-core` and `networkx` instead of `cirq`.
- Tests: 32 → 47.

## 0.3.0 — 2026-09-21

### Fixed (correctness)
- `decompose_fredkin_to_sycamore` returned a circuit containing a raw 3-qubit `CCXPowGate`; the transformer now expands ≥3-qubit gates before synthesis.
- `SycamoreCalibrationMap` was never used by the compiler despite the README; `decompose(..., calibration=...)` and the transformer's `calibration_map` now freeze the FSim to the coupler's native angles.
- NumPy fallback gradient used `Tr(·)²` instead of `|Tr(·)|²`.
- OpenQASM 3 export defined `gate fsim` with an empty body (an identity), emitted no `defcal`, silently dropped Y/S/T/CNOT/CZ/PhasedXZ gates, and ordered qubits by string (`q(10)` before `q(2)`). All fixed; a parser enables round-trip tests.
- "Riemannian" optimisation was L-BFGS over Euler angles; a real manifold solver is now available as `method="riemannian"`, and the Euler-angle solver is called what it is.

### Added
- `FSimCircuitTemplate(fixed_fsim=...)`, `CouplerCalibration.sycamore()/sqrt_iswap()`, calibration-map drift, `to_dict/from_dict`.
- `RiemannianFSimSolver` with Cayley retractions and JAX Lie-algebra gradients.
- Seeded multi-start from a native-point library (`SEED_ANGLES`), loss history, `haar_random_unitary`, `euler_zxz_from_unitary`.
- `@cirq.transformer compile_to_fsim`, decomposition cache, transformer statistics.
- CCZ and 3-qubit QFT decompositions, `qft_unitary`.
- `export_to_openqasm3(expand_fsim=True)`, `parse_openqasm3`.
- Batch compiler threads and stage histogram; CLI `--native/--calibrated/--method`, `fredkin|ccz|qft3`, `qasm`, `benchmark`, `--json`.
- 32 tests (from 14); browser demo runs the synthesiser for real.

### Facts established by the tests
- Free angles: every named 2-qubit gate = 1 FSim (SWAP = FSim(π/2, π) up to local Z); Haar-random U(4) = 2 FSims.
- Native Sycamore FSim(π/2, π/6): CZ = 2, SWAP = 3, random U(4) = 3. √iSWAP: iSWAP = 2, not 1.

## 0.2.0
- Calibration map, Toffoli/Fredkin, OpenQASM 3 export, batch compiler, CLI, CI, notebook.

## 0.1.0
- Initial release.
