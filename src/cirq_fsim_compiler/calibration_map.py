"""
Google Sycamore Hardware Calibration Map and Coupler Drift Ingestion.

Allows the compiler to synthesize circuits directly onto measured, drift-adjusted
physical coupler parameters (theta_cal, phi_cal) for each individual physical edge (q_i, q_j).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any


@dataclass
class CouplerCalibration:
    """Stores calibrated native parameters for a specific physical coupler."""
    theta_cal: float = np.pi / 2.0  # Nominal ~ 1.5708
    phi_cal: float = np.pi / 6.0    # Nominal ~ 0.5236
    depolarizing_error: float = 0.004
    cz_phase_error: float = 0.01


class SycamoreCalibrationMap:
    """
    Database of hardware calibration parameters across physical grid couplers.
    """

    def __init__(self, seed: Optional[int] = None):
        self.couplers: Dict[Tuple[int, int], CouplerCalibration] = {}
        self.rng = np.random.default_rng(seed)

    def set_coupler(self, q0: int, q1: int, calibration: CouplerCalibration):
        key = (min(q0, q1), max(q0, q1))
        self.couplers[key] = calibration

    def get_coupler(self, q0: int, q1: int) -> CouplerCalibration:
        key = (min(q0, q1), max(q0, q1))
        if key not in self.couplers:
            # Default with slight physical Gaussian drift
            d_theta = float(self.rng.normal(0.0, 0.02))
            d_phi = float(self.rng.normal(0.0, 0.015))
            self.couplers[key] = CouplerCalibration(
                theta_cal=np.pi / 2.0 + d_theta,
                phi_cal=np.pi / 6.0 + d_phi,
            )
        return self.couplers[key]
