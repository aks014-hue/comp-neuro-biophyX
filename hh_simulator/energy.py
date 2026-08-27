"""Molecular-scale energy landscape for ion-channel gating.

Each gating particle (m, h, n) is modeled as a two-state system (closed <-> open)
whose voltage-dependent transition rates are derived from Eyring rate theory
(transition-state theory).

Parameterization (primary molecular parameters):
  k0      : prefactor / attempt rate (1/ms). Sets the peak time constant
            tau_max = 1/(2*k0) at V = V_half.
  z       : gating charge in elementary charges (SIGNED).
            z > 0  -> activation particle (x_inf increases with V; e.g. m, n)
            z < 0  -> inactivation particle (x_inf decreases with V; e.g. h)
  V_half  : half-activation voltage (mV). The voltage where x_inf = 0.5.
            Encodes the energy asymmetry: dG0 = z * e * V_half.
  delta   : symmetry factor (0..1). Fraction of the gating charge that has
            moved at the transition state. delta = 0.5 gives a symmetric,
            bell-shaped tau(V).

Derived energy-landscape quantities (for visualization / interpretation):
  dG0       = z * e * V_half            (open-vs-closed energy difference at V=0)
  dG_dagger = -kT * ln(k0 / A)          (baseline activation barrier; A = attempt freq)

Rate functions (Eyring):
  alpha(V) = k0 * exp( delta      * z * (V - V_half) / V_T )   [closed -> open]
  beta(V)  = k0 * exp( -(1-delta) * z * (V - V_half) / V_T )   [open  -> closed]

Steady state and time constant:
  x_inf(V) = alpha / (alpha + beta) = 1 / (1 + exp(-z*(V - V_half)/V_T))   (Boltzmann)
  tau(V)   = 1 / (alpha + beta)                                            (bell-shaped)

A two-state Eyring model reproduces the Boltzmann-shaped x_inf(V) and bell-shaped
tau(V) of classic HH well, but does NOT exactly equal the empirical HH rate
functions (which use the (V+a)/(1-exp(-(V+a)/b)) form). The fitted preset
approximates classic-HH x_inf/tau; the exact empirical rates are available as a
separate "classic" mode (see presets.py / channels.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from .units import thermal_voltage, E_CHARGE, K_B, ATTEMPT_FREQ


@dataclass
class EnergyLandscape:
    """Two-state Eyring energy landscape for a single gating particle.

    Optional rate caps (alpha_cap, beta_cap) impose a maximum transition rate,
    physically motivated by the Kramers limit: a rate cannot exceed the
    molecular attempt frequency, equivalent to a minimum activation barrier.
    Without caps, Eyring rates grow exponentially and tau collapses at extreme
    voltages; classic HH rates saturate (e.g. beta_h -> 1). The caps let the
    Eyring model reproduce that saturation, which is essential for action-potential
    generation.
    """

    k0: float            # prefactor, 1/ms
    z: float             # signed gating charge, elementary charges
    V_half: float        # half-activation voltage, mV
    delta: float = 0.5   # symmetry factor, 0..1
    temp_c: float = 6.3  # temperature, degrees C
    name: str = ""       # particle label (m, h, n, ...)
    alpha_cap: float = float("inf")  # max closed->open rate, 1/ms (Kramers limit)
    beta_cap: float = float("inf")   # max open->closed rate, 1/ms

    # ---- derived quantities ----
    @property
    def V_T(self) -> float:
        return thermal_voltage(self.temp_c)

    @property
    def dG0(self) -> float:
        """Open-vs-closed energy difference at V=0, in kT units."""
        return self.z * self.V_half / self.V_T

    @property
    def dG_dagger(self) -> float:
        """Baseline activation barrier in kT units (from k0 and attempt freq)."""
        return -np.log(self.k0 / ATTEMPT_FREQ)

    # ---- rate functions ----
    def alpha(self, V) -> np.ndarray:
        V = np.asarray(V, dtype=float)
        arg = self.delta * self.z * (V - self.V_half) / self.V_T
        rate = self.k0 * np.exp(np.clip(arg, -700, 700))
        return np.minimum(rate, self.alpha_cap)

    def beta(self, V) -> np.ndarray:
        V = np.asarray(V, dtype=float)
        arg = -(1.0 - self.delta) * self.z * (V - self.V_half) / self.V_T
        rate = self.k0 * np.exp(np.clip(arg, -700, 700))
        return np.minimum(rate, self.beta_cap)

    def x_inf(self, V) -> np.ndarray:
        V = np.asarray(V, dtype=float)
        return 1.0 / (1.0 + np.exp(np.clip(-self.z * (V - self.V_half) / self.V_T,
                                           -700, 700)))

    def tau(self, V) -> np.ndarray:
        a = self.alpha(V)
        b = self.beta(V)
        return 1.0 / (a + b)

    # ---- barrier-vs-voltage (for energy-landscape plots) ----
    def barrier_closed(self, V) -> np.ndarray:
        """Activation barrier from the closed state, in kT units."""
        V = np.asarray(V, dtype=float)
        return self.dG_dagger - self.delta * self.z * (V - self.V_half) / self.V_T

    def barrier_open(self, V) -> np.ndarray:
        """Activation barrier from the open state, in kT units."""
        V = np.asarray(V, dtype=float)
        return self.dG_dagger + (1.0 - self.delta) * self.z * (V - self.V_half) / self.V_T

    def __repr__(self) -> str:
        return (f"EnergyLandscape(name={self.name!r}, k0={self.k0:.4g}/ms, "
                f"z={self.z:.3g}, V_half={self.V_half:.3g}mV, delta={self.delta:.3g})")


# ------------------------------------------------------------------
# Fitting a two-state Eyring landscape to classic-HH steady state / tau
# ------------------------------------------------------------------
def _classic_x_inf_tau(alpha_fn: Callable, beta_fn: Callable, V: np.ndarray):
    a = np.asarray(alpha_fn(V), dtype=float)
    b = np.asarray(beta_fn(V), dtype=float)
    xinf = a / (a + b)
    tau = 1.0 / (a + b)
    return xinf, tau


def fit_to_classic(alpha_fn: Callable, beta_fn: Callable,
                   V_range: Optional[np.ndarray] = None,
                   temp_c: float = 6.3, name: str = "",
                   delta: Optional[float] = None) -> EnergyLandscape:
    """Fit {k0, z, V_half, delta} so the Eyring x_inf/tau best-match classic HH.

    Strategy:
      V_half  <- voltage where classic x_inf = 0.5 (interpolated)
      z       <- slope of the Boltzmann logit at V_half  (from x_inf curvature)
      k0      <- 1 / (2 * tau_max), where tau_max is the peak classic tau
      delta   <- fit by 1-D scan minimizing tau-shape error (default 0.5)
    """
    if V_range is None:
        V_range = np.linspace(-100.0, 50.0, 601)
    V = np.asarray(V_range, dtype=float)
    xinf_c, tau_c = _classic_x_inf_tau(alpha_fn, beta_fn, V)

    # --- V_half: where x_inf crosses 0.5 ---
    # find sign change of (xinf - 0.5)
    d = xinf_c - 0.5
    sign = np.sign(d)
    # handle monotonic; find first crossing
    crossings = np.where(np.diff(sign) != 0)[0]
    if len(crossings) == 0:
        V_half = V[np.argmin(np.abs(d))]
    else:
        i = crossings[0]
        # linear interpolation between V[i] and V[i+1]
        t = d[i] / (d[i] - d[i + 1])
        V_half = V[i] + t * (V[i + 1] - V[i])

    # --- z: from the Boltzmann slope at V_half ---
    # logit(x_inf) = z*(V - V_half)/V_T  =>  z = V_T * d(logit)/dV
    # use a robust finite-difference of logit away from saturation
    eps = 5.0  # mV window
    mask = (np.abs(xinf_c - 0.5) < 0.35) & (xinf_c > 1e-4) & (xinf_c < 1 - 1e-4)
    if mask.sum() >= 2:
        logit = np.log(xinf_c / (1.0 - xinf_c))
        z = np.polyfit(V[mask], logit[mask], 1)[0] * thermal_voltage(temp_c)
    else:
        z = 4.0  # fallback
    # preserve sign: activation z>0, inactivation z<0
    # determine from monotonic direction of x_inf
    if xinf_c[-1] < xinf_c[0]:
        z = -abs(z)
    else:
        z = abs(z)

    # --- k0: from peak tau ---
    tau_max = np.nanmax(tau_c[np.isfinite(tau_c)])
    k0 = 1.0 / (2.0 * tau_max)

    # --- delta: 1-D scan minimizing tau shape error ---
    def tau_err(d_):
        el = EnergyLandscape(k0=k0, z=z, V_half=V_half, delta=d_, temp_c=temp_c)
        t_e = el.tau(V)
        # normalize both to peak=1 to compare shape
        denom = np.nanmax(tau_c[np.isfinite(tau_c)])
        return np.nansum((t_e / (2 * k0) ** -1 - tau_c / denom) ** 2) if False else \
            np.nansum((el.tau(V) / np.nanmax(el.tau(V)) -
                       tau_c / np.nanmax(tau_c[np.isfinite(tau_c)])) ** 2)

    if delta is None:
        ds = np.linspace(0.2, 0.8, 31)
        errs = np.array([tau_err(d_) for d_ in ds])
        delta = float(ds[np.argmin(errs)])
    else:
        delta = float(delta)

    # --- rate caps (Kramers limit): prevent tau collapse at extreme voltages ---
    # Classic HH rates saturate where Eyring rates grow exponentially.
    # For activation (z>0) alpha grows -> cap alpha; for inactivation (z<0)
    # beta grows -> cap beta.  Cap = max classic rate over the fit range.
    a_c = np.asarray(alpha_fn(V), dtype=float)
    b_c = np.asarray(beta_fn(V), dtype=float)
    if z > 0:
        alpha_cap = float(np.nanmax(a_c[np.isfinite(a_c)]))
        beta_cap = float("inf")
    else:
        alpha_cap = float("inf")
        beta_cap = float(np.nanmax(b_c[np.isfinite(b_c)]))

    return EnergyLandscape(k0=k0, z=z, V_half=V_half, delta=delta,
                           temp_c=temp_c, name=name,
                           alpha_cap=alpha_cap, beta_cap=beta_cap)
