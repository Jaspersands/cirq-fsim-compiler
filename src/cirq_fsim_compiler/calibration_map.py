"""
Calibration data for physical couplers.

A ``CouplerCalibration`` holds the measured native FSim angles (θ_cal, φ_cal)
of one coupler; ``SycamoreCalibrationMap`` stores them per edge and can be
perturbed with Gaussian drift to emulate day-to-day calibration changes. The
synthesiser consumes these through ``decompose(..., calibration=...)`` so the
compiled circuit uses the gate the hardware actually implements.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple

import numpy as np

SYCAMORE_THETA = np.pi / 2.0
SYCAMORE_PHI = np.pi / 6.0
SQRT_ISWAP_THETA = np.pi / 4.0


@dataclass
class CouplerCalibration:
    """Native FSim(θ_cal, φ_cal) of a coupler plus error figures."""

    theta_cal: float = SYCAMORE_THETA
    phi_cal: float = SYCAMORE_PHI
    depolarizing_error: float = 0.004
    cz_phase_error: float = 0.01

    @property
    def angles(self) -> Tuple[float, float]:
        return (float(self.theta_cal), float(self.phi_cal))

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    @classmethod
    def sycamore(cls) -> "CouplerCalibration":
        return cls(theta_cal=SYCAMORE_THETA, phi_cal=SYCAMORE_PHI)

    @classmethod
    def sqrt_iswap(cls) -> "CouplerCalibration":
        return cls(theta_cal=SQRT_ISWAP_THETA, phi_cal=0.0)


class SycamoreCalibrationMap:
    """Per-edge calibration store; unknown edges get the nominal gate plus drift."""

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
