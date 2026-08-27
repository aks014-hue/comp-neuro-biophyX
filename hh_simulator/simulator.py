"""Simulator: deterministic ODE and stochastic single-channel modes (switchable).

Deterministic mode integrates the full HH ODE system with scipy.solve_ivp
(adaptive stepping; RK45 default, BDF option for stiff regimes).

Stochastic mode simulates each gating-particle pool as an independent two-state
Markov process via binomial tau-leaping at a fixed dt (the standard "stochastic
Hodgkin-Huxley" approach, e.g. Chow & White 1996; Schneidman et al. 1998). The
membrane voltage is advanced with an unconditionally-stable exponential Euler
step. Total channel counts per channel type are user-settable; as counts grow
the stochastic trajectory converges to the deterministic one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Union

import numpy as np
from scipy.integrate import solve_ivp

from .cell import PointCell


@dataclass
class Solution:
    """Container for a simulation result."""
    t: np.ndarray
    V: np.ndarray
    gating: Dict[str, np.ndarray]      # particle name -> trajectory
    currents: Dict[str, np.ndarray]    # channel name -> current trajectory
    mode: str
    metadata: Dict = field(default_factory=dict)

    @property
    def dt(self) -> np.ndarray:
        return np.diff(self.t)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _make_I_fn(I_inj: Union[float, Callable]) -> Callable:
    if callable(I_inj):
        return I_inj
    val = float(I_inj)
    return lambda t: val


def _step_current(t0: float, t1: float, amp: float, onset: float, dur: float) -> Callable:
    """Rectangular current pulse: amp for onset <= t < onset+dur, else 0."""
    def I(t):
        return amp if (onset <= t < onset + dur) else 0.0
    return I


# ------------------------------------------------------------------
# Deterministic mode
# ------------------------------------------------------------------
def _run_deterministic(cell: PointCell, t_span, I_inj, t_eval,
                       method, rtol, atol, V0, **kwargs):
    parts = cell.particles                     # [(channel, particle), ...]
    pnames = [p.name for (_, p) in parts]
    I_fn = _make_I_fn(I_inj)

    if V0 is None:
        V0, state0 = cell.initial_state()
    else:
        state0 = cell.steady_state(V0)

    y0 = [float(V0)] + [float(state0[n]) for n in pnames]

    def rhs(t, y):
        V = y[0]
        st = {pnames[i]: y[i + 1] for i in range(len(pnames))}
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            dV = cell.dVdt(V, st, I_fn(t))
            dy = [dV]
            for i, (ch, p) in enumerate(parts):
                a = float(np.asarray(p.alpha(V)).reshape(-1)[0])
                b = float(np.asarray(p.beta(V)).reshape(-1)[0])
                x = y[i + 1]
                dy.append(a * (1.0 - x) - b * x)
        return dy

    sol = solve_ivp(rhs, t_span, y0, method=method, t_eval=t_eval,
                    rtol=rtol, atol=atol, dense_output=t_eval is None, **kwargs)
    if t_eval is None:
        t = sol.t
    else:
        t = np.asarray(t_eval, dtype=float)

    V = sol.sol(t)[0] if t_eval is None else sol.y[0]
    gating = {}
    for i, name in enumerate(pnames):
        gating[name] = sol.sol(t)[i + 1] if t_eval is None else sol.y[i + 1]

    # per-channel currents
    currents = {ch.name: np.zeros_like(t) for ch in cell.channels}
    for k, tk in enumerate(t):
        st = {name: gating[name][k] for name in pnames}
        for ch in cell.channels:
            currents[ch.name][k] = ch.current(V[k], st)

    I_trace = np.array([I_fn(tk) for tk in t])
    return Solution(t=t, V=np.asarray(V), gating=gating, currents=currents,
                    mode="deterministic",
                    metadata=dict(method=method, I_inj=I_trace, V0=float(V0)))


# ------------------------------------------------------------------
# Stochastic mode
# ------------------------------------------------------------------
def _run_stochastic(cell: PointCell, t_span, I_inj, dt, N_channels,
                    rng, t_eval, V0, record_every, **kwargs):
    parts = cell.particles
    pnames = [p.name for (_, p) in parts]
    I_fn = _make_I_fn(I_inj)

    if V0 is None:
        V0, state0 = cell.initial_state()
    else:
        state0 = cell.steady_state(V0)
    V = float(V0)

    # Build particle pools: name -> (N_total, N_open, particle, channel)
    pools = {}
    for ch, p in parts:
        N_ch = int(N_channels.get(ch.name, 0))
        N_total = p.power * N_ch
        N_open = int(round(state0[p.name] * N_total))
        pools[p.name] = dict(N_total=N_total, N_open=N_open, p=p, ch=ch, N_ch=N_ch)

    t0, tf = float(t_span[0]), float(t_span[1])
    n_steps = int(np.ceil((tf - t0) / dt))
    ts = t0 + np.arange(n_steps + 1) * dt
    ts[-1] = min(ts[-1], tf)

    # storage
    rec_V = []
    rec_t = []
    rec_gating = {name: [] for name in pnames}
    rec_currents = {ch.name: [] for ch in cell.channels}

    def record(t_cur):
        rec_t.append(t_cur)
        rec_V.append(V)
        frac = {}
        for name in pnames:
            pl = pools[name]
            f = pl["N_open"] / pl["N_total"] if pl["N_total"] > 0 else state0[name]
            frac[name] = f
            rec_gating[name].append(f)
        st = frac
        for ch in cell.channels:
            rec_currents[ch.name].append(ch.current(V, st))

    record(t0)
    rec_idx = 0

    for s in range(1, n_steps + 1):
        t_cur = ts[s]
        # --- update particle pools (binomial tau-leaping) ---
        for name in pnames:
            pl = pools[name]
            if pl["N_total"] <= 0:
                continue
            p = pl["p"]
            a = float(np.asarray(p.alpha(V)).reshape(-1)[0])
            b = float(np.asarray(p.beta(V)).reshape(-1)[0])
            p_open = 1.0 - np.exp(-a * dt)    # closed -> open prob
            p_close = 1.0 - np.exp(-b * dt)   # open -> closed prob
            n_open = pl["N_open"]
            n_closed = pl["N_total"] - n_open
            to_open = rng.binomial(n_closed, p_open)
            to_close = rng.binomial(n_open, p_close)
            pl["N_open"] = n_open + to_open - to_close

        # --- fractions & effective conductances ---
        frac = {name: (pools[name]["N_open"] / pools[name]["N_total"]
                       if pools[name]["N_total"] > 0 else state0[name])
                for name in pnames}

        g_eff = 0.0
        gE = 0.0
        for ch in cell.channels:
            if ch.is_gated:
                f = 1.0
                for p in ch.particles:
                    f *= frac[p.name] ** p.power
            else:
                f = 1.0
            g_eff += ch.g_bar * f
            gE += ch.g_bar * f * ch.E

        Ij = I_fn(t_cur)
        # exponential Euler: C_m dV/dt = -g_eff*(V - E_eff) + I_inj
        if g_eff > 0:
            E_eff = gE / g_eff
            V_star = E_eff + Ij / g_eff
            V = V_star + (V - V_star) * np.exp(-g_eff * dt / cell.C_m)
        else:
            V = V + (Ij / cell.C_m) * dt

        if s % record_every == 0 or s == n_steps:
            record(t_cur)

    t_arr = np.array(rec_t)
    V_arr = np.array(rec_V)
    gating = {name: np.array(rec_gating[name]) for name in pnames}
    currents = {name: np.array(rec_currents[name]) for name in rec_currents}

    # subsample to t_eval if requested
    if t_eval is not None:
        t_eval = np.asarray(t_eval, dtype=float)
        idx = np.array([np.argmin(np.abs(t_arr - te)) for te in t_eval])
        t_arr = t_arr[idx]
        V_arr = V_arr[idx]
        gating = {k: v[idx] for k, v in gating.items()}
        currents = {k: v[idx] for k, v in currents.items()}

    I_trace = np.array([I_fn(tk) for tk in t_arr])
    return Solution(t=t_arr, V=V_arr, gating=gating, currents=currents,
                    mode="stochastic",
                    metadata=dict(dt=dt, N_channels=dict(N_channels),
                                  I_inj=I_trace, V0=float(V0)))


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------
class Simulator:
    """Wraps a PointCell and runs deterministic or stochastic simulations."""

    def __init__(self, cell: PointCell):
        self.cell = cell

    def run(self, t_span, I_inj=0.0, mode="deterministic", t_eval=None,
            method="RK45", rtol=1e-8, atol=1e-8, V0=None,
            dt=0.01, N_channels=None, rng=None, record_every=1, **kwargs):
        if mode == "deterministic":
            return _run_deterministic(self.cell, t_span, I_inj, t_eval,
                                      method, rtol, atol, V0, **kwargs)
        elif mode == "stochastic":
            if N_channels is None:
                N_channels = {"Na": 1000, "K": 300}
            if rng is None:
                rng = np.random.default_rng()
            return _run_stochastic(self.cell, t_span, I_inj, dt, N_channels,
                                   rng, t_eval, V0, record_every, **kwargs)
        else:
            raise ValueError(f"Unknown mode {mode!r}; use 'deterministic' or 'stochastic'.")


def run(cell: PointCell, t_span, I_inj=0.0, mode="deterministic", **kwargs) -> Solution:
    """Convenience function: run a simulation on a cell."""
    return Simulator(cell).run(t_span, I_inj=I_inj, mode=mode, **kwargs)


def step_pulse(t_span, amp, onset=None, dur=None):
    """Build a rectangular current-pulse callable for I_inj.

    Defaults: onset = 10% of span, dur = 80% of span.
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    if onset is None:
        onset = t0 + 0.1 * (tf - t0)
    if dur is None:
        dur = 0.8 * (tf - t0)
    return _step_current(t0, tf, amp, onset, dur)
