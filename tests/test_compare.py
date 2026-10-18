"""
Tests for the comparison with Cirq: baselines, the circuit suite, the
noise-aware bookkeeping and the new CLI subcommands.
"""

import json

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")

from cirq_fsim_compiler import compare as C  # noqa: E402
from cirq_fsim_compiler.cli import main as cli_main  # noqa: E402
from cirq_fsim_compiler.toffoli_decomposer import qft_unitary  # noqa: E402


def fid(u, v):
    return abs(np.trace(u.conj().T @ v)) ** 2 / u.shape[0] ** 2


def test_cirq_baseline_counts():
    # Sycamore, cirq-core: controlled-phase classes (CNOT, CZ, CPhase) take two gates through
    # KAK + decompose_cphase_into_two_fsim, everything else four
    expect = {  # gate: (cz, sqrt_iswap, sycamore)
        "CNOT": (1, 2, 2), "CZ": (1, 2, 2), "iSWAP": (2, 2, 4), "SWAP": (3, 3, 4), "CPhase(1π/8)": (2, 2, 2),
    }
    targets = {t["name"]: t for t in C.two_qubit_targets(n_haar=0)}
    for name, (n_cz, n_sq, n_syc) in expect.items():
        t = targets[name]
        for gs, n in (("cz", n_cz), ("sqrt_iswap", n_sq), ("sycamore", n_syc)):
            ops, _ = C.cirq_two_qubit(t["unitary"], gs, t["gate"])
            assert C._two_qubit_count(ops) == n, (name, gs)
            q = cirq.LineQubit.range(2)
            assert C.process_infidelity(t["unitary"], C._ops_unitary(ops, q)) < 1e-12
    assert C.cirq_two_qubit(np.eye(4), "sycamore")[1] == "local"
    ops, routine = C.cirq_two_qubit(targets["SWAP"]["unitary"], "free")
    assert C._two_qubit_count(ops) == 3 and routine == "cz"           # ties go to full CZs
    ops, routine = C.cirq_two_qubit(targets["√iSWAP"]["unitary"], "free")
    assert C._two_qubit_count(ops) == 1 and routine == "sqrt_iswap"
    ops, routine = C.cirq_two_qubit(targets["CPhase(1π/8)"]["unitary"], "free")
    assert C._two_qubit_count(ops) == 1 and routine == "cz_partial"   # a partial CZ is FSim(0, φ)


def test_cirq_google_baseline_uses_the_named_gate():
    cirq_google = pytest.importorskip("cirq_google")
    gs = cirq_google.SycamoreTargetGateset()
    targets = {t["name"]: t for t in C.two_qubit_targets(n_haar=0)}
    q = cirq.LineQubit.range(2)
    for name in ("SWAP", "iSWAP"):
        t = targets[name]
        ops = C.cirq_google_two_qubit(t["unitary"], gs, t["gate"])
        assert C._two_qubit_count(ops) == 3, name
        assert C.process_infidelity(t["unitary"], C._ops_unitary(ops, q)) < 1e-9


@pytest.mark.slow
def test_compare_two_qubit_matches_cirq_on_analytic_gate_sets():
    targets = [t for t in C.two_qubit_targets(n_haar=2, seed=5) if t["name"] in ("CNOT", "SWAP", "√iSWAP") or t["group"] == "Haar"]
    rows = C.compare_two_qubit(targets, gate_sets=("cz", "sqrt_iswap"), include_cirq_google=False,
                               unpruned=lambda t: t["group"] == "Haar")
    assert len(rows) == 2 * len(targets)
    for r in rows:
        assert r["ours_n"] == r["cirq_n"] == r["min_n"] and r["min_kind"] == "closed form"
        assert r["ours_infidelity"] < 1e-8 and r["cirq_infidelity"] < 1e-12
    summ = C.summarize_two_qubit(rows)
    assert summ["cz"]["at_minimum"] == 1.0 and summ["cz"]["unpruned_targets"] == 2
    md = C.markdown_tables(rows)
    assert "Haar-random U(4)" in md and "closed form 5" in md


def test_noise_aware_choice_and_sweep():
    profiles = [[0.5, 0.1, 1e-3, 0.0], [0.5, 0.2, 0.05, 0.0]]
    assert C.noise_aware_choice(profiles[0], 3e-3) == (2, pytest.approx(7e-3), pytest.approx(9e-3))
    assert C.noise_aware_choice(profiles[1], 3e-3)[0] == 3
    sw = C.noise_aware_sweep(profiles, [1e-5, 3e-3, 1.0])
    assert sw["fraction_fewer"] == [0.0, 0.5, 1.0]
    assert sw["mean_saved"][1] == pytest.approx(1e-3)
    assert all(s >= 0 for s in sw["mean_saved"])


def test_circuit_suite_builders():
    q = cirq.LineQubit.range(4)
    for n in (3, 4):
        assert fid(cirq.unitary(C.qft_circuit(q[:n])), qft_unitary(n)) > 1 - 1e-12
    adder = C.cuccaro_adder_circuit(*q)          # (c, b, a, z) -> (c, a ⊕ b ⊕ c, a, z ⊕ carry)
    sim = cirq.Simulator()
    for bits in range(8):
        c, b, a = (bits >> 2) & 1, (bits >> 1) & 1, bits & 1
        prep = cirq.Circuit([cirq.X(qb) for qb, v in zip(q[:3], (c, b, a)) if v])
        out = sim.run(prep + adder + cirq.Circuit(cirq.measure(*q, key="m"))).measurements["m"][0].tolist()
        assert out == [c, (a + b + c) % 2, a, int(a + b + c >= 2)]
    ghz = cirq.final_state_vector(C.ghz_circuit(q))
    assert np.allclose(np.abs(ghz[[0, 15]]) ** 2, 0.5)
    suite = C.circuit_suite()
    assert len(suite) == 11 and all(len(c.all_qubits()) in (3, 4) for c in suite.values())


def test_compare_circuits_small():
    suite = {k: v for k, v in C.circuit_suite().items() if k in ("GHZ (4q)", "Toffoli")}
    rows = C.compare_circuits(suite, gate_sets=("cz", "sycamore"), include_cirq_google=False)
    for r in rows:
        assert r["cirq_F"] > 1 - 1e-9 and r["ours_F"] > 1 - 1e-7 and r["ours_failures"] == 0
        assert 1 - r["na_F"] <= r["na_bound"] + 1e-9
        assert r["na_est"] >= r["ours_est"] - 1e-9
        assert r["ours_n"] <= r["cirq_n"]
    by = {(r["circuit"], r["gate_set"]): r for r in rows}
    assert by["GHZ (4q)", "sycamore"]["cirq_n"] == 6                 # two per CNOT through the cphase route
    s = C.summarize_circuits(rows)
    assert s["sycamore"]["ours_total_2q"] <= s["sycamore"]["cirq_total_2q"] and s["cz"]["na_bound_holds"]


def test_cli_weyl_and_synthesize_cz(capsys):
    assert cli_main(["weyl", "--gate", "iswap", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert np.allclose(out["weyl_coordinates"], [np.pi / 4, np.pi / 4, 0])
    assert (out["min_cz"], out["min_sqrt_iswap"], out["min_free_fsim"], out["min_sycamore"]) == (2, 2, 1, 3)
    assert out["min_sycamore_kind"] == "proven"                       # two Sycamores reach only x + y ≤ 5π/12
    assert cli_main(["synthesize", "--gate", "swap", "--target", "cz", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_stages"] == 3 and out["native"] is True


def test_cli_compare_subcommand(capsys, monkeypatch):
    calls = {}

    def fake_run(**kw):
        calls.update(kw)
        return {"two_qubit_summary": {}, "noise": {"at_cited": {}}, "circuit_summary": None, "markdown": "table"}
    monkeypatch.setattr(C, "run_comparison", fake_run)
    assert cli_main(["compare", "--quick", "--no-circuits", "--json"]) == 0
    assert calls["quick"] is True and calls["circuits"] is False
    assert "two_qubit_summary" in json.loads(capsys.readouterr().out)


def test_markdown_labels_and_compact_results(tmp_path):
    targets = [t for t in C.two_qubit_targets(n_haar=1, seed=3, cphase_steps=4) if t["group"] in ("CPhase", "Haar")]
    rows = [{"target": t["name"], "group": t["group"], "gate_set": "cz", "ours_n": 2, "ours_infidelity": 0.0,
             "ours_s": 0.01, "cirq_n": 2, "cirq_infidelity": 0.0, "cirq_s": 0.001, "cirq_routine": "cz",
             "min_n": 2, "min_kind": "closed form"} for t in targets]
    md = C.markdown_tables(rows)
    assert "CPhase(kπ/4), k = 1..3" in md and "kπ/8" not in md
    res = {"meta": {}, "two_qubit": rows, "two_qubit_summary": C.summarize_two_qubit(rows),
           "group_counts": C.group_counts(rows), "haar_weyl": [[0.1234567, 0.0, 0.0]],
           "noise": {"eps_grid": [1e-3], "sweep": {}, "at_cited": {}, "check": {}, "profiles": {"cz": [[0.123456789, 0.0]]}},
           "markdown": md}
    small = C.compact_results(res)
    assert small["profiles"]["cz"] == [[0.1235, 0.0]] and len(small["two_qubit_named"]) == 3
    path = tmp_path / "r.json"
    C.dump_json(small, str(path))
    assert json.loads(path.read_text())["timing"]["cz"]["ours_ms"]
