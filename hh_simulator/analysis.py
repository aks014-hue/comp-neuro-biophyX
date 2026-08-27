"""Analysis: action-potential detection, F-I curves, gating dynamics, rate profiles."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import find_peaks

from .cell import PointCell
from .simulator import Simulator, Solution, step_pulse
from . import presets


def detect_aps(V: np.ndarray, t: np.ndarray, threshold: float = 0.0,
               refractory: float = 2.0) -> Dict:
    """Detect action potentials by peak finding above a threshold.

    Returns dict with spike_times, peaks (mV), n_spikes, and mean_rate (Hz)
    over the full trace.
    """
    V = np.asarray(V, dtype=float)
    t = np.asarray(t, dtype=float)
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    dist = max(int(refractory / dt), 1)
    peaks_idx, props = find_peaks(V, height=threshold, distance=dist)
    spike_times = t[peaks_idx]
    peak_vals = V[peaks_idx]
    duration_s = (t[-1] - t[0]) / 1000.0
    mean_rate = len(spike_times) / duration_s if duration_s > 0 else 0.0
    return dict(spike_times=spike_times, peaks=peak_vals, indices=peaks_idx,
                n_spikes=len(spike_times), mean_rate=mean_rate,
                threshold=threshold)


def fi_curve(simulator: Simulator, I_range: np.ndarray,
             t_span: Tuple[float, float] = (0.0, 500.0),
             onset: float = 50.0, steady_window: Tuple[float, float] = (150.0, 500.0),
             mode: str = "deterministic", threshold: float = 0.0,
             **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """Compute firing rate vs injected current (F-I curve).

    For each current amplitude, inject a step pulse and count APs in the
    steady-state window. Returns (I, rate_Hz).
    """
    t0, tf = t_span
    ws, we = steady_window
    rates = []
    for I in I_range:
        I_fn = step_pulse(t_span, I, onset=onset, dur=tf - onset)
        sol = simulator.run(t_span, I_inj=I_fn, mode=mode, **kwargs)
        aps = detect_aps(sol.V, sol.t, threshold=threshold)
        st = aps["spike_times"]
        mask = (st >= ws) & (st <= we)
        n = int(mask.sum())
        rate = n / ((we - ws) / 1000.0)
        rates.append(rate)
    return np.asarray(I_range, dtype=float), np.array(rates, dtype=float)


def gating_dynamics(solution: Solution) -> Dict[str, np.ndarray]:
    """Return per-particle gating trajectories from a Solution."""
    return dict(solution.gating)


def steady_state_rates(cell: PointCell, V_range: np.ndarray) -> Dict:
    """Compute alpha, beta, x_inf, tau for every particle over V_range."""
    out = {}
    for ch in cell.channels:
        for p in ch.particles:
            a = p.alpha(V_range)
            b = p.beta(V_range)
            out[p.name] = dict(alpha=a, beta=b, x_inf=a / (a + b), tau=1.0 / (a + b))
    return out


def classic_steady_state_rates(V_range: np.ndarray) -> Dict:
    """alpha/beta/x_inf/tau for the empirical classic-HH particles."""
    out = {}
    for name, (af, bf) in presets.CLASSIC_RATES.items():
        a = np.asarray(af(V_range), dtype=float)
        b = np.asarray(bf(V_range), dtype=float)
        out[name] = dict(alpha=a, beta=b, x_inf=a / (a + b), tau=1.0 / (a + b))
    return out
