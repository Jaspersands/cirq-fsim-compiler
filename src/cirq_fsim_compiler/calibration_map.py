"""
Calibration data for physical couplers.

A ``CouplerCalibration`` holds the measured native FSim angles (θ_cal, φ_cal)
of one coupler; ``SycamoreCalibrationMap`` stores them per edge and can be
perturbed with Gaussian drift to emulate day-to-day calibration changes. The
synthesiser consumes these through ``decompose(..., calibration=...)`` so the
compiled circuit uses the gate the hardware implements.

Three gate families have constructors:

* ``CouplerCalibration.sycamore()``: FSim(π/2, π/6), Google Sycamore.
* ``CouplerCalibration.sqrt_iswap()``: FSim(π/4, 0). This equals cirq.SQRT_ISWAP_INV
  and is locally equivalent to cirq.SQRT_ISWAP.
* ``CouplerCalibration.cz()``: FSim(0, π). With this package's (and Cirq's)
  convention FSim(θ, φ)[3, 3] = e^{−iφ}, FSim(0, π) = diag(1, 1, 1, −1) is
  exactly cirq.CZ, not only up to single-qubit gates. CZ is the native gate
  on Google Willow and on neutral-atom processors.

``gate_error`` is the process infidelity of one native two-qubit gate, used
by noise-aware synthesis. The constructors set it from published figures,
converted where needed. For two qubits (d = 4), an average gate infidelity
r corresponds to a process infidelity of (d + 1)/d · r = 5/4 · r.

* Sycamore: 0.62%, the mean simultaneous two-qubit Pauli error in Arute et
  al., Nature 574, 505 (2019). A Pauli error is already a process
  infidelity, so it is used as is.
* Willow CZ: 0.41%. The Willow spec sheet (Google Quantum AI, 2024) gives a
  mean simultaneous CZ error of 0.33% from randomized benchmarking, an
  average gate infidelity; 5/4 · 0.33% = 0.41%.
* √iSWAP: 0.86%. Huang et al., PRL 130, 070601 (2023) report a mean SQiSW
  fidelity of 99.31% from interleaved fully randomized benchmarking, an
  average gate fidelity; 5/4 · 0.69% = 0.86%.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

SYCAMORE_THETA = np.pi / 2.0
SYCAMORE_PHI = np.pi / 6.0
SQRT_ISWAP_THETA = np.pi / 4.0
CZ_THETA = 0.0
CZ_PHI = np.pi

#: Published two-qubit gate errors as process infidelities (sources and conversions in the module docstring).
SYCAMORE_GATE_ERROR = 0.0062
WILLOW_CZ_ERROR = 1.25 * 0.0033
SQRT_ISWAP_GATE_ERROR = 1.25 * 0.0069


@dataclass
class CouplerCalibration:
    """Native FSim(θ_cal, φ_cal) of a coupler plus error figures."""

    theta_cal: float = SYCAMORE_THETA
    phi_cal: float = SYCAMORE_PHI
    depolarizing_error: float = 0.004
    cz_phase_error: float = 0.01
    gate_error: Optional[float] = None

    @property
    def angles(self) -> Tuple[float, float]:
        return (float(self.theta_cal), float(self.phi_cal))

    def to_dict(self) -> Dict[str, Optional[float]]:
        return asdict(self)

    @classmethod
    def sycamore(cls) -> "CouplerCalibration":
        return cls(theta_cal=SYCAMORE_THETA, phi_cal=SYCAMORE_PHI, gate_error=SYCAMORE_GATE_ERROR)

    @classmethod
    def sqrt_iswap(cls) -> "CouplerCalibration":
        return cls(theta_cal=SQRT_ISWAP_THETA, phi_cal=0.0, gate_error=SQRT_ISWAP_GATE_ERROR)

    @classmethod
    def cz(cls) -> "CouplerCalibration":
        """FSim(0, π), which equals cirq.CZ exactly."""
        return cls(theta_cal=CZ_THETA, phi_cal=CZ_PHI, gate_error=WILLOW_CZ_ERROR)


class SycamoreCalibrationMap:
    """
    Per-edge calibration store; unknown edges get the nominal gate plus drift.

    Despite the name, ``nominal`` can be any calibration, for example
    ``CouplerCalibration.cz()`` with zero drift for a CZ device.
    """

    def __init__(self, seed: Optional[int] = None, theta_drift_std: float = 0.02, phi_drift_std: float = 0.015,
                 nominal: Optional[CouplerCalibration] = None):
        self.couplers: Dict[Tuple[int, int], CouplerCalibration] = {}
        self.rng = np.random.default_rng(seed)
        self.theta_drift_std = float(theta_drift_std)
        self.phi_drift_std = float(phi_drift_std)
        self.nominal = nominal or CouplerCalibration.sycamore()

    @staticmethod
    def _key(q0: int, q1: int) -> Tuple[int, int]:
        return (min(q0, q1), max(q0, q1))

    def set_coupler(self, q0: int, q1: int, calibration: CouplerCalibration) -> None:
        self.couplers[self._key(q0, q1)] = calibration

    def get_coupler(self, q0: int, q1: int) -> CouplerCalibration:
        key = self._key(q0, q1)
        if key not in self.couplers:
            self.couplers[key] = CouplerCalibration(
                theta_cal=self.nominal.theta_cal + float(self.rng.normal(0.0, self.theta_drift_std)),
                phi_cal=self.nominal.phi_cal + float(self.rng.normal(0.0, self.phi_drift_std)),
                depolarizing_error=self.nominal.depolarizing_error,
                cz_phase_error=self.nominal.cz_phase_error,
                gate_error=self.nominal.gate_error,
            )
        return self.couplers[key]

    def perturb(self, seed: Optional[int] = None) -> "SycamoreCalibrationMap":
        """Return a copy with fresh Gaussian drift applied to every stored coupler."""
        rng = np.random.default_rng(seed)
        new = SycamoreCalibrationMap(seed=seed, theta_drift_std=self.theta_drift_std, phi_drift_std=self.phi_drift_std, nominal=self.nominal)
        for key, cal in self.couplers.items():
            new.couplers[key] = CouplerCalibration(
                theta_cal=cal.theta_cal + float(rng.normal(0.0, self.theta_drift_std)),
                phi_cal=cal.phi_cal + float(rng.normal(0.0, self.phi_drift_std)),
                depolarizing_error=cal.depolarizing_error, cz_phase_error=cal.cz_phase_error,
                gate_error=cal.gate_error,
            )
        return new

    def to_dict(self) -> Dict[str, Any]:
        return {f"{a}-{b}": cal.to_dict() for (a, b), cal in sorted(self.couplers.items())}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SycamoreCalibrationMap":
        cm = cls()
        for key, val in data.items():
            a, b = (int(x) for x in key.split("-"))
            cm.couplers[cm._key(a, b)] = CouplerCalibration(**val)
        return cm
