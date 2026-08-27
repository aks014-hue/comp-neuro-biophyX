"""Ion channels (channel scale) for the classic HH model.

Each channel is composed of gating particles. A particle's voltage-dependent
transition rates come from either:
  - an EnergyLandscape (Eyring rate theory, "energy" mode), or
  - exact empirical classic-HH alpha/beta callables ("classic" mode).

Conductance follows the classic HH stoichiometry: Na ~ m^3 * h, K ~ n^4, leak
is ungated. Current sign convention: outward current positive
(I = g_bar * gating * (V - E)).
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Union

import numpy as np

from . import presets
from .energy import EnergyLandscape


class GatingParticle:
    """A single gating particle with a voltage-dependent rate source."""

    def __init__(self, name: str, power: int,
                 energy: Optional[EnergyLandscape] = None,
                 alpha_fn: Optional[Callable] = None,
                 beta_fn: Optional[Callable] = None):
        self.name = name
        self.power = power
        self.energy = energy
        if energy is not None:
            self._alpha_fn = energy.alpha
            self._beta_fn = energy.beta
        elif alpha_fn is not None and beta_fn is not None:
            self._alpha_fn = alpha_fn
            self._beta_fn = beta_fn
        else:
            raise ValueError("Provide either `energy` or both `alpha_fn` and `beta_fn`.")

    def alpha(self, V):
        return np.asarray(self._alpha_fn(V), dtype=float)

    def beta(self, V):
        return np.asarray(self._beta_fn(V), dtype=float)

    def x_inf(self, V):
        a = self.alpha(V)
        b = self.beta(V)
        return a / (a + b)

    def tau(self, V):
        a = self.alpha(V)
        b = self.beta(V)
        return 1.0 / (a + b)

    def __repr__(self) -> str:
        src = "energy" if self.energy is not None else "classic"
        return f"GatingParticle({self.name!r}, power={self.power}, src={src})"


class IonChannel:
    """Base ion channel: max conductance, reversal, and gating particles."""

    def __init__(self, name: str, g_bar: float, E: float,
                 particles: List[GatingParticle]):
        self.name = name
        self.g_bar = g_bar
        self.E = E
        self.particles = particles

    @property
    def is_gated(self) -> bool:
        return len(self.particles) > 0

    def conductance_factor(self, state: Dict[str, float]) -> float:
        """Fractional conductance from gating-variable values (0..1)."""
        f = 1.0
        for p in self.particles:
            f *= state[p.name] ** p.power
        return f

    def current(self, V: float, state: Dict[str, float]) -> float:
        """Ionic current (outward positive), uA/cm^2."""
        return self.g_bar * self.conductance_factor(state) * (V - self.E)

    def steady_state(self, V: float) -> Dict[str, float]:
        return {p.name: float(p.x_inf(V)) for p in self.particles}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(g_bar={self.g_bar}, E={self.E}, particles={self.particles})"


class NaChannel(IonChannel):
    """Voltage-gated sodium channel: conductance ~ m^3 * h."""

    def __init__(self, mode: str = "classic",
                 m_energy: Optional[EnergyLandscape] = None,
                 h_energy: Optional[EnergyLandscape] = None,
                 g_bar: Optional[float] = None, E: Optional[float] = None):
        if m_energy is not None and h_energy is not None:
            m = GatingParticle("m", 3, energy=m_energy)
            h = GatingParticle("h", 1, energy=h_energy)
        elif mode == "classic":
            m = GatingParticle("m", 3, alpha_fn=presets.alpha_m, beta_fn=presets.beta_m)
            h = GatingParticle("h", 1, alpha_fn=presets.alpha_h, beta_fn=presets.beta_h)
        elif mode == "energy":
            m = GatingParticle("m", 3, energy=presets.fitted_landscape("m"))
            h = GatingParticle("h", 1, energy=presets.fitted_landscape("h"))
        else:
            raise ValueError(f"Unknown mode {mode!r}; use 'classic' or 'energy'.")
        super().__init__("Na", g_bar if g_bar is not None else presets.G_NA,
                         E if E is not None else presets.E_NA, [m, h])


class KChannel(IonChannel):
    """Delayed-rectifier potassium channel: conductance ~ n^4."""

    def __init__(self, mode: str = "classic",
                 n_energy: Optional[EnergyLandscape] = None,
                 g_bar: Optional[float] = None, E: Optional[float] = None):
        if n_energy is not None:
            n = GatingParticle("n", 4, energy=n_energy)
        elif mode == "classic":
            n = GatingParticle("n", 4, alpha_fn=presets.alpha_n, beta_fn=presets.beta_n)
        elif mode == "energy":
            n = GatingParticle("n", 4, energy=presets.fitted_landscape("n"))
        else:
            raise ValueError(f"Unknown mode {mode!r}; use 'classic' or 'energy'.")
        super().__init__("K", g_bar if g_bar is not None else presets.G_K,
                         E if E is not None else presets.E_K, [n])


class LeakChannel(IonChannel):
    """Ungated leak conductance."""

    def __init__(self, g_bar: Optional[float] = None, E: Optional[float] = None):
        super().__init__("Leak", g_bar if g_bar is not None else presets.G_L,
                         E if E is not None else presets.E_L, [])

    def conductance_factor(self, state) -> float:
        return 1.0


def classic_cell() -> "PointCell":
    """Convenience: build a classic-HH point cell (imported lazily to avoid cycle)."""
    from .cell import PointCell
    return PointCell([NaChannel("classic"), KChannel("classic"), LeakChannel()])
