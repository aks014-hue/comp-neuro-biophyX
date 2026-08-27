"""Classic Hodgkin-Huxley parameters (squid axon, 6.3 C) and fitted Eyring presets.

Rate functions use the modern absolute-mV convention (resting potential ~ -65 mV).
Removable singularities in alpha_n (V = -55 mV) and alpha_m (V = -40 mV) are
handled with a series-limit safe evaluation.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from .units import V_T_6_3
from .energy import EnergyLandscape, fit_to_classic

# ------------------------------------------------------------------
# Classic HH conductances, reversals, capacitance, temperature
# ------------------------------------------------------------------
G_NA = 120.0    # mS/cm^2
G_K = 36.0      # mS/cm^2
G_L = 0.3       # mS/cm^2
E_NA = 50.0     # mV
E_K = -77.0     # mV
E_L = -54.4     # mV
C_M = 1.0       # uF/cm^2
TEMP_C = 6.3    # degrees C

# ------------------------------------------------------------------
# Classic HH empirical rate functions (absolute-mV convention)
# ------------------------------------------------------------------
def _safe_exp(x):
    """exp with argument clipped to avoid overflow/underflow warnings."""
    return np.exp(np.clip(x, -700.0, 700.0))


def _safe_ratio(V, a, b, scale):
    """scale * (V+a) / (1 - exp(-(V+a)/b)) with removable-singularity handling.

    Limit as (V+a) -> 0 of x/(1-exp(-x/b)) = b, so the limit of the full
    expression is scale * b. For small |x| we use the first-order series
    x/(1-exp(-x/b)) ~ b + x/2 to stay accurate near the singularity.
    """
    V = np.asarray(V, dtype=float)
    x = V + a
    small = np.abs(x) < 1e-4
    # series: b + x/2  (valid to O(x^2)); evaluate only where needed
    series = b + x / 2.0
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        direct = x / (1.0 - _safe_exp(-x / b))
    val = np.where(small, series, direct)
    if val.ndim == 0:
        val = float(val)
    return scale * val


def alpha_n(V):
    return _safe_ratio(V, 55.0, 10.0, 0.01)

def beta_n(V):
    V = np.asarray(V, dtype=float)
    return 0.125 * _safe_exp(-(V + 65.0) / 80.0)

def alpha_m(V):
    return _safe_ratio(V, 40.0, 10.0, 0.1)

def beta_m(V):
    V = np.asarray(V, dtype=float)
    return 4.0 * _safe_exp(-(V + 65.0) / 18.0)

def alpha_h(V):
    V = np.asarray(V, dtype=float)
    return 0.07 * _safe_exp(-(V + 65.0) / 20.0)

def beta_h(V):
    V = np.asarray(V, dtype=float)
    return 1.0 / (1.0 + _safe_exp(-(V + 35.0) / 10.0))


# Particle -> (alpha, beta) callables
CLASSIC_RATES: Dict[str, Tuple[Callable, Callable]] = {
    "m": (alpha_m, beta_m),
    "h": (alpha_h, beta_h),
    "n": (alpha_n, beta_n),
}

# Particle -> gating-power stoichiometry
STOICHIOMETRY: Dict[str, int] = {"m": 3, "h": 1, "n": 4}


# ------------------------------------------------------------------
# Fitted Eyring energy-landscape presets (computed lazily on first access)
# ------------------------------------------------------------------
_FITTED_CACHE: Dict[str, EnergyLandscape] = {}


def fitted_landscape(particle: str, V_range: np.ndarray = None) -> EnergyLandscape:
    """Return the Eyring EnergyLandscape fitted to classic HH for a particle."""
    if particle not in _FITTED_CACHE:
        a, b = CLASSIC_RATES[particle]
        _FITTED_CACHE[particle] = fit_to_classic(
            a, b, V_range=V_range, temp_c=TEMP_C, name=particle)
    return _FITTED_CACHE[particle]


def fitted_presets() -> Dict[str, EnergyLandscape]:
    """Return fitted Eyring landscapes for all three particles."""
    return {p: fitted_landscape(p) for p in ("m", "h", "n")}


def classic_params() -> dict:
    """Return the classic HH parameter dict."""
    return dict(
        g_na=G_NA, g_k=G_K, g_l=G_L,
        e_na=E_NA, e_k=E_K, e_l=E_L,
        c_m=C_M, temp_c=TEMP_C,
    )
