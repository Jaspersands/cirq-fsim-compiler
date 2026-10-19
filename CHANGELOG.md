# Changelog

## 0.5.0 — 2026-10-19

Results measured against Cirq's built-in decompositions, exact Weyl-chamber gate counts for every fixed FSim, a CZ target and noise-aware approximate synthesis.

### Added
- `weyl` module: canonical coordinates from `cirq.kak_decomposition`; closed-form minimum counts for CZ, √iSWAP (Huang et al. 2023) and free-angle FSim, for a unitary or any coordinate 3-vector (canonicalised first); the closed-form best fidelity with zero or one native gate; `predict_stages`; `numerical_min_count`.
- Controlled-phase strengths and the lower bounds they give (`cphase_strength`, `cphase_count_excluded`): n controlled phases reach only x + y + |z| ≤ n·s, and with an odd number of FSim(π/2, φ) the class needs x + y + |z| ≥ 3π/4 − n·s. On Sycamore, two gates reach only z = 0 with x + y ≤ 5π/12, so iSWAP provably needs three, and classes with z ≠ 0 and x + y + |z| < π/8 need four. The three-gate bound is attained at (π/24, π/24, π/24).
- `monodromy` module: the exact region reachable with n copies of a fixed two-qubit gate and any single-qubit gates between them, from the monodromy polytope of Peterson, Crooks and Smith (2020). The quantum Littlewood–Richardson table is copied from Peterson's `monodromy` package (Apache-2.0), which is not a dependency because it needs `lrs`; the n-gate regions are chained with 2ⁿ⁻¹ linear programs in scipy. For FSim(0, φ) and FSim(π/2, φ) the closed form of Peterson, Bishop and Javadi-Abhari (2022) is used instead. `weyl.n_gate_region_contains` gives the test for any FSim.
- `weyl.haar_density`: the Haar density on the Weyl chamber (Watts, O'Connor and Vala 2013).
- `minimum_count`: the minimum and how it is known ("closed form", or "proven" from the exact regions; "not found" above `max_stages`). `method="search"` runs the multi-start search instead, which can also answer "numerical". The CLI `weyl` command reports it for Sycamore.
- `benchmarks/fsim_regions.py` and `figures/fsim_regions.png`: the Haar measure of the Sycamore four-gate region, and minimum-count shares on drifted couplers, with a synthesis check of each.
- Up to four fixed native gates by default (`max_stages=4`; templates accept up to six, and larger values raise `ValueError`), with Cirq's four-FSim routine as a fallback at four, since three Sycamore gates do not cover the chamber.
- For controlled-phase classes, `cirq.decompose_cphase_into_two_fsim` after a KAK change of basis is tried at two gates on every calibration; Cirq's routine decides whether the native gate can make the controlled phase.
- `SynthesisWarning`, `SynthesisError` and `on_failure` ("warn", "raise", "ignore") on the synthesiser and the transformer. The test suite turns the warning into an error.
- `CouplerCalibration.cz()`: FSim(0, π), which equals `cirq.CZ` exactly with the shared FSim convention.
- Noise-aware synthesis, `approximation="noise_aware"` on the synthesiser, the transformer and `cirq-fsim synthesize --noise-aware EPS`: choose n minimising (1 − F(n)) + n·ε. Zero gates, one fixed gate and one free FSim use closed forms; other fits use 16 restarts × 500 iterations. The transformer records block infidelities, and `infidelity_bound()` bounds the whole circuit's approximation error.
- `compare` module, `benchmarks/compare_with_cirq.py`, `benchmarks/make_figures.py`, `figures/`, and the CLI subcommands `compare` and `weyl`. The circuit suite includes Toffoli, Fredkin, CCZ, QFT3, QFT4, a Cuccaro adder, GHZ and seeded random circuits. `tabulation_study` compares Cirq's approximate `two_qubit_gate_product_tabulation` route with noise-aware synthesis.
- Zero-stage decompositions for local targets, `calibration=` on the transformer, `--target {sycamore,sqrt_iswap,cz}` on the CLI.

### Fixed
- An exact synthesis that missed its tolerance was returned without notice. It now issues a `SynthesisWarning` (or raises `SynthesisError` with `on_failure="raise"`). Noise-aware mode reports nothing, since an approximation was asked for.
- Measurements were moved to the end of the circuit. A measurement followed by other operations on its qubit now raises `ValueError`; terminal measurements and resets are kept in place.
- Classically controlled operations and non-unitary channels were not rejected. They now raise `ValueError`.
- An input `FSimGate` was kept whatever its angles. It is now kept only if its angles match the calibration of its edge; `PhasedFSimGate` and other angles are resynthesised.

### Changed
- `gate_error` values are process infidelities: Sycamore 0.62% (a Pauli error, unchanged), √iSWAP 0.86% and Willow CZ 0.41% (5/4 of the published average gate infidelities 0.69% and 0.33%).
- Baselines: `cirq_google.SycamoreTargetGateset` is given the named gate and is the main Sycamore baseline; the cirq-core Sycamore baseline uses the controlled-phase route for every y = z = 0 class (CNOT takes two) in both the two-qubit and the circuit comparison; the free-FSim baseline includes partial CZs.
- Timing is the best of three calls for every routine, this package's included. Circuits are timed once per route after a warm-up.
- The stage search starts at the minimum count from the Weyl coordinates, which is exact for every fixed FSim. When it fails there it retries once with three times the restarts and five times the iterations.
- `predict_stages` uses the exact regions by default (`polytope=False` gives the hand-proved lower bounds only), and `StagePrediction.source` says which ("closed form", "polytope", "lower bound").
- The Weyl-chamber tests use a coordinate tolerance of √ε for a target infidelity ε; the zero- and one-gate tests use the closed-form infidelity.
- Compiled JAX objectives are cached per process. In the Euclidean solver the angles of a fixed native gate are an argument, so each stage count compiles once whatever the calibration.
- The committed benchmark output is compact (about 100 KB); the full output goes to `benchmarks/results/full/`, which is ignored.
- `requirements.txt` matches `pyproject.toml` (the stale PennyLane entry is gone).
- Tests: 47 to 144. The `slow` marker covers the 11 tests that take more than about 1.5 s. The suite took about 4 minutes on a laptop that was running other jobs.

### Findings
- Sycamore: three gates reach exactly the classes with x + y + |z| ≥ π/8, so the classes that need four are those with z ≠ 0 and x + y + |z| < π/8, and no others. Their Haar measure is 1.3746 × 10⁻⁴, about 1 in 7,300 (4 × 10⁶ Haar samples give 1.39 ± 0.06 × 10⁻⁴).
- A drift in θ makes the two-gate region three-dimensional: with φ = π/6, 2.55% of Haar-random targets need two gates at |θ − π/2| = 0.005, 24.1% at 0.05 and 90.65% at 0.3. On the calibrated pair (3,3)–(3,4) of the supremacy experiment, FSim(1.2947, 0.4859), 88.8% need two. Multi-start synthesis agrees with the polytope on 90 of 90 checked targets.
- Two-qubit targets (5 named gates, CPhase(kπ/8) for k = 1 to 7, 200 Haar-random unitaries): every result is at the minimum count, and every Sycamore minimum is proven. On CZ and √iSWAP the counts equal Cirq's. On Sycamore a Haar-random target takes 3 gates, against 4 from the cirq-core routine and 6 from `cirq_google`; on the named gates `cirq_google` matches this package except √iSWAP (4 against 2). Median time per target: 6.4 to 11 ms for this package, 0.7 to 3.6 ms for the cirq-core routines and 7.9 ms for `cirq_google`.
- Circuit suite (11 circuits, 79 blocks), all compiled equivalently in exact mode: CZ 126 / 126 and √iSWAP 159 / 159 (Cirq / this package). Sycamore 202 (`cirq_google`), 192 (cirq-core) and 171 (this package); mean estimated fidelity F · (1 − ε)ⁿ 0.894, 0.899 and 0.909.
- Noise-aware at the cited ε on Haar-random targets: fewer gates are better for 43% on Sycamore, 36% on CZ, 26.5% with free angles and 17.5% on √iSWAP. The fits are upper bounds, so these are lower bounds; a rerun with another seed and twice the restarts agreed on 40 of 40 targets per gate set and found no lower cost. For CZ the third gate is dropped exactly when sin²z < ε. Cirq's tabulation route on Sycamore averages 2.83 gates and an expected infidelity of 1.85e-2, against 1.67e-2 for noise-aware synthesis and 1.86e-2 for exact synthesis. On the circuit suite noise-aware synthesis changes 1 to 3 blocks per gate set (126 to 125 CZ, 159 to 157 √iSWAP, 171 to 167 Sycamore), including Coppersmith's approximate QFT in QFT4, and stays within the bound.

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
