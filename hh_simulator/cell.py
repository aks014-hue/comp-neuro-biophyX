"""Point cell (single isopotential compartment) for the HH model.

Membrane equation:
  C_m * dV/dt = -sum(I_ionic) + I_inj
where I_ionic = g_bar * gating * (V - E) for each channel (outward positive).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.optimize import brentq

from .channels import IonChannel
from . import presets


class PointCell:
    """A single-compartment isopotential cell holding a set of ion channels."""

    def __init__(self, channels: List[IonChannel], C_m: float = presets.C_M,
                 temp_c: float = presets.TEMP_C):
        self.channels = channels
        self.C_m = C_m
        self.temp_c = temp_c

    @property
    def particle_names(self) -> List[str]:
        names: List[str] = []
        for ch in self.channels:
            for p in ch.particles:
                names.append(p.name)
        return names

    @property
    def particles(self):
        """Flat list of (channel, particle) pairs in canonical order."""
        out = []
        for ch in self.channels:
            for p in ch.particles:
                out.append((ch, p))
        return out

    def steady_state(self, V: float) -> Dict[str, float]:
        state: Dict[str, float] = {}
        for ch in self.channels:
            for p in ch.particles:
                state[p.name] = float(p.x_inf(V))
        return state

    def total_ionic_current(self, V: float, state: Dict[str, float]) -> float:
        I = 0.0
        for ch in self.channels:
            I += ch.current(V, state)
        return I

    def channel_currents(self, V: float, state: Dict[str, float]) -> Dict[str, float]:
        return {ch.name: ch.current(V, state) for ch in self.channels}

    def dVdt(self, V: float, state: Dict[str, float], I_inj: float) -> float:
        return (-self.total_ionic_current(V, state) + I_inj) / self.C_m

    def resting_potential(self, V_lo: float = -100.0, V_hi: float = 40.0) -> float:
        """Voltage where steady-state ionic current is zero (no injection)."""
        def f(V):
            return self.total_ionic_current(V, self.steady_state(V))
        return float(brentq(f, V_lo, V_hi))

    def initial_state(self, V0: float = None) -> tuple:
        """Return (V0, steady-state gating dict) for simulation start."""
        if V0 is None:
            V0 = self.resting_potential()
        return float(V0), self.steady_state(V0)

    def __repr__(self) -> str:
        return (f"PointCell(channels={[c.name for c in self.channels]}, "
                f"C_m={self.C_m}, temp_c={self.temp_c})")
