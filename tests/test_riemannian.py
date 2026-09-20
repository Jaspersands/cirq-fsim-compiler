"""
Tests for the Riemannian (Cayley-retraction) solver.
"""

import numpy as np
import pytest

cirq = pytest.importorskip("cirq")

from cirq_fsim_compiler.riemannian_manifold import RiemannianFSimSolver, cayley_retract, lie_element
from cirq_fsim_compiler.riemannian_optimizer import DifferentiableFSimSynthesizer
from cirq_fsim_compiler.calibration_map import CouplerCalibration
from cirq_fsim_compiler.unitary_ansatz import haar_random_unitary, FSimCircuitTemplate


def test_cayley_retraction_is_exactly_unitary():
    rng = np.random.default_rng(0)
    V = haar_random_unitary(2, seed=1)
    for _ in range(20):
        A = lie_element(rng.normal(size=4) * 3.0)
        W = cayley_retract(V, A)
        assert np.allclose(W.conj().T @ W, np.eye(2), atol=1e-13)


def test_riemannian_gradient_matches_finite_difference():
    solver = RiemannianFSimSolver(n_stages=2, seed=0)
    target = cirq.unitary(cirq.CNOT)
    blocks = [haar_random_unitary(2, seed=k) for k in range(solver.n_blocks)]
    angles = np.array([0.3, 0.7, 1.1, -0.4])
    g = solver.gradient(blocks, angles, target)
    # finite-difference reference (the solver's own fallback path)
    solver._grad = None
    g_fd = solver.gradient(blocks, angles, target)
    assert np.allclose(g, g_fd, rtol=1e-4, atol=1e-6)


def test_riemannian_solver_reaches_high_fidelity_and_stays_unitary():
    solver = RiemannianFSimSolver(n_stages=2, max_iter=800, seed=3)
    res = solver.solve(cirq.unitary(cirq.CNOT), n_restarts=4, target_infidelity=1e-9)
    assert res.method == "riemannian"
    assert res.infidelity < 1e-8
    assert np.allclose(res.synthesized_unitary.conj().T @ res.synthesized_unitary, np.eye(4), atol=1e-12)
    assert all(h2 <= h1 + 1e-15 for h1, h2 in zip(res.loss_history, res.loss_history[1:]))  # monotone
    # extracted angles rebuild the same unitary through the Euler-angle template
    template = FSimCircuitTemplate(n_stages=2)
    u_rebuilt = template.evaluate_unitary_np(res.optimal_params)
    assert abs(np.trace(u_rebuilt.conj().T @ res.synthesized_unitary)) ** 2 / 16 > 1 - 1e-9


def test_riemannian_and_euclidean_agree():
    u = haar_random_unitary(4, seed=21)
    e = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=1, method="euclidean").decompose(u, n_restarts=6, max_iter=400)
    r = DifferentiableFSimSynthesizer(target_infidelity=1e-9, seed=1, method="riemannian").decompose(u, n_restarts=6, max_iter=800)
    assert e.infidelity < 1e-8 and r.infidelity < 1e-6
    assert abs(e.fidelity - r.fidelity) < 1e-6


def test_riemannian_calibrated_mode():
    cal = CouplerCalibration.sycamore()
    r = DifferentiableFSimSynthesizer(target_infidelity=1e-8, seed=2, method="riemannian").decompose(
        cirq.unitary(cirq.CZ), n_restarts=6, max_iter=800, calibration=cal)
    assert r.native and r.n_stages == 2 and r.infidelity < 1e-7
    assert all(np.allclose(a, cal.angles) for a in r.fsim_angles)
