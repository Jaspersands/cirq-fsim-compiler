Two-qubit gate counts (mean over the group; minimum in brackets)

| target | free FSim: ours / Cirq | Sycamore: ours / cirq_google / Cirq core | √iSWAP: ours / Cirq | CZ: ours / Cirq |
|---|---|---|---|---|
| CNOT | 1 / 1 [1] | 2 / 2 / 2 [2] | 2 / 2 [2] | 1 / 1 [1] |
| CZ | 1 / 1 [1] | 2 / 2 / 2 [2] | 2 / 2 [2] | 1 / 1 [1] |
| iSWAP | 1 / 2 [1] | 3 / 3 / 4 [3] | 2 / 2 [2] | 2 / 2 [2] |
| √iSWAP | 1 / 1 [1] | 2 / 4 / 4 [2] | 1 / 1 [1] | 2 / 2 [2] |
| SWAP | 1 / 3 [1] | 3 / 3 / 4 [3] | 3 / 3 [3] | 3 / 3 [3] |
| CPhase(kπ/8), k = 1..7 | 1 / 1 [1] | 2 / 2 / 2 [2] | 2 / 2 [2] | 2 / 2 [2] |
| Haar-random U(4) | 2 / 2.21 [2] | 3 / 6 / 4 [3] | 2.21 / 2.21 [2.21] | 3 / 3 [3] |

Accuracy, speed and minimality over all targets (times: best of three calls)

| gate set | at minimum (without pruning) | how the minimum is known | process infidelity, ours (median / max) | Cirq (max) | ms per decomposition, ours / Cirq | without Weyl pruning (ms) | speed-up |
|---|---|---|---|---|---|---|---|
| free FSim | 100.0% (99.5%) | closed form 212 | 4e-16 / 9e-15 | 1e-15 | 6 / 3.6 | 29 | 4.4× |
| Sycamore | 100.0% (100.0%) | proven 212 | 4e-16 / 9e-15 | 2e-15 (cirq_google), 1e-15 (core) | 8 / 7.9 (cirq_google), 2.7 (core) | 58 | 6.7× |
| √iSWAP | 100.0% (99.1%) | closed form 212 | 9e-16 / 1e-12 | 2e-15 | 11 / 0.7 | 41 | 2.7× |
| CZ | 100.0% (100.0%) | closed form 212 | 4e-16 / 6e-15 | 1e-15 | 8 / 1.4 | 55 | 6.7× |

Whole circuits: two-qubit gate count

| circuit | CZ: Cirq / ours / noise-aware | √iSWAP: Cirq / ours / noise-aware | Sycamore: cirq_google / Cirq core / ours / noise-aware |
|---|---|---|---|
| Toffoli | 6 / 6 / 6 | 10 / 10 / 10 | 10 / 10 / 10 / 10 |
| Fredkin | 7 / 7 / 7 | 12 / 12 / 12 | 12 / 12 / 12 / 12 |
| CCZ | 6 / 6 / 6 | 10 / 10 / 10 | 10 / 10 / 10 / 10 |
| QFT3 | 9 / 9 / 9 | 9 / 9 / 9 | 9 / 10 / 9 / 9 |
| QFT4 | 18 / 18 / 18 | 18 / 18 / 16 | 18 / 20 / 18 / 16 |
| Adder (4q) | 21 / 21 / 21 | 42 / 42 / 42 | 42 / 42 / 42 / 42 |
| GHZ (4q) | 3 / 3 / 3 | 6 / 6 / 6 | 6 / 6 / 6 / 6 |
| random 3q #1 | 9 / 9 / 9 | 10 / 10 / 10 | 13 / 14 / 11 / 11 |
| random 3q #2 | 10 / 10 / 10 | 11 / 11 / 11 | 16 / 16 / 13 / 13 |
| random 4q #1 | 18 / 18 / 18 | 16 / 16 / 16 | 34 / 26 / 21 / 20 |
| random 4q #2 | 19 / 19 / 18 | 15 / 15 / 15 | 32 / 26 / 19 / 18 |
| total | 126 / 126 / 125 | 159 / 159 / 157 | 202 / 192 / 171 / 167 |

Whole circuits: estimated fidelity F · (1 − ε)^n

| circuit | CZ: Cirq / ours / noise-aware | √iSWAP: Cirq / ours / noise-aware | Sycamore: cirq_google / Cirq core / ours / noise-aware |
|---|---|---|---|
| Toffoli | 0.976 / 0.976 / 0.976 | 0.917 / 0.917 / 0.917 | 0.940 / 0.940 / 0.940 / 0.940 |
| Fredkin | 0.971 / 0.971 / 0.971 | 0.901 / 0.901 / 0.901 | 0.928 / 0.928 / 0.928 / 0.928 |
| CCZ | 0.976 / 0.976 / 0.976 | 0.917 / 0.917 / 0.917 | 0.940 / 0.940 / 0.940 / 0.940 |
| QFT3 | 0.963 / 0.963 / 0.963 | 0.925 / 0.925 / 0.925 | 0.946 / 0.940 / 0.946 / 0.946 |
| QFT4 | 0.928 / 0.928 / 0.928 | 0.856 / 0.856 / 0.862 | 0.894 / 0.883 / 0.894 / 0.897 |
| Adder (4q) | 0.917 / 0.917 / 0.917 | 0.695 / 0.695 / 0.695 | 0.770 / 0.770 / 0.770 / 0.770 |
| GHZ (4q) | 0.988 / 0.988 / 0.988 | 0.949 / 0.949 / 0.949 | 0.963 / 0.963 / 0.963 / 0.963 |
| random 3q #1 | 0.963 / 0.963 / 0.963 | 0.917 / 0.917 / 0.917 | 0.922 / 0.917 / 0.934 / 0.934 |
| random 3q #2 | 0.960 / 0.960 / 0.960 | 0.909 / 0.909 / 0.909 | 0.905 / 0.905 / 0.922 / 0.922 |
| random 4q #1 | 0.928 / 0.928 / 0.928 | 0.871 / 0.871 / 0.871 | 0.809 / 0.851 / 0.878 / 0.879 |
| random 4q #2 | 0.924 / 0.924 / 0.928 | 0.878 / 0.878 / 0.878 | 0.820 / 0.851 / 0.889 / 0.894 |
| mean | 0.954 / 0.954 / 0.954 | 0.885 / 0.885 / 0.886 | 0.894 / 0.899 / 0.909 / 0.910 |

Noise-aware synthesis on Haar-random U(4): fraction of targets where the best fit found with fewer gates has lower (1 − F) + n·ε than the exact decomposition

| gate set | ε (process infidelity) | fewer gates better | mean saving (all) | mean saving (when fewer) | rerun check: same n / lower cost found |
|---|---|---|---|---|---|
| free FSim | 0.62% | 26.5% | 8.36e-04 | 3.15e-03 | 40/40 / 0 |
| Sycamore | 0.62% | 43.0% | 1.89e-03 | 4.39e-03 | 40/40 / 0 |
| √iSWAP | 0.86% | 17.5% | 1.04e-03 | 5.94e-03 | 40/40 / 0 |
| CZ | 0.41% | 36.0% | 1.13e-03 | 3.13e-03 | 40/40 / 0 |

Sycamore, approximate routes at ε = 0.62% (mean over 200 Haar-random targets)

| route | mean gates | mean synthesis infidelity | mean (1 − F) + n·ε |
|---|---|---|---|
| Cirq tabulation (max_infidelity 0.01) | 2.83 | 9.19e-04 | 1.85e-02 |
| noise-aware synthesis | 2.57 | – | 1.67e-02 |
| exact synthesis | – | – | 1.86e-02 |
