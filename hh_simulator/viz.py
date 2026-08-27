"""Visualization: voltage traces, gating dynamics, F-I curves, energy landscapes,
rate profiles. All figures saved as SVG (editable text) + PNG.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .energy import EnergyLandscape
from .simulator import Solution
from . import presets, analysis

# ---- global plot style ----
matplotlib.rcParams["font.family"] = ["Liberation Sans", "Arimo", "DejaVu Sans"]
matplotlib.rcParams["svg.fonttype"] = "none"        # keep text editable in SVG
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False

# colorblind-friendly palette (subset of Phylo palette + safe extras)
PALETTE = {
    "V": "#000000",
    "Na": "#0279EE",     # blue
    "K": "#FF9400",      # orange
    "Leak": "#75A025",   # green
    "m": "#0279EE",
    "h": "#FD9BED",      # pink
    "n": "#FF9400",
    "classic": "#000000",
    "eyring": "#E9ED4C",  # yellow-green
    "inj": "#888888",
}
PARTICLE_LABELS = {"m": "m (Na activation)", "h": "h (Na inactivation)",
                   "n": "n (K activation)"}


def _save(fig, savepath: Optional[str]):
    if savepath is None:
        return
    fig.savefig(savepath, bbox_inches="tight")
    if savepath.endswith(".svg"):
        png = savepath[:-4] + ".png"
    else:
        png = savepath + ".png"
    fig.savefig(png, bbox_inches="tight", dpi=150)


def plot_voltage(solution: Solution, ax=None, savepath: Optional[str] = None,
                 title: str = "Membrane potential"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    else:
        fig = ax.figure
    ax.plot(solution.t, solution.V, color=PALETTE["V"], lw=1.2, label="V")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, savepath)
    return fig, ax


def plot_gating(solution: Solution, ax=None, savepath: Optional[str] = None,
                title: str = "Gating variables"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    else:
        fig = ax.figure
    for name, traj in solution.gating.items():
        ax.plot(solution.t, traj, color=PALETTE.get(name, "#333333"),
                lw=1.2, label=PARTICLE_LABELS.get(name, name))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Gating variable")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, savepath)
    return fig, ax


def plot_currents(solution: Solution, ax=None, savepath: Optional[str] = None,
                  title: str = "Ionic currents"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.2))
    else:
        fig = ax.figure
    for name, traj in solution.currents.items():
        ax.plot(solution.t, traj, color=PALETTE.get(name, "#333333"),
                lw=1.2, label=name)
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Current (uA/cm^2)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, savepath)
    return fig, ax


def plot_fi_curve(I: np.ndarray, rate: np.ndarray, ax=None,
                  savepath: Optional[str] = None, label: str = "",
                  title: str = "F-I curve"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
    else:
        fig = ax.figure
    ax.plot(I, rate, "o-", color=PALETTE["Na"], lw=1.5, ms=4, label=label)
    ax.set_xlabel("Injected current (uA/cm^2)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title)
    if label:
        ax.legend(frameon=False)
    _save(fig, savepath)
    return fig, ax


def plot_rates(particle: str, V_range: np.ndarray, ax=None,
               savepath: Optional[str] = None,
               title: Optional[str] = None):
    """Plot alpha, beta, x_inf, tau vs V for a particle, Eyring vs classic overlay."""
    if ax is None:
        fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    else:
        fig = ax.figure
        axes = ax
    if title is None:
        title = f"Particle {particle}: Eyring fit vs classic HH"

    el = presets.fitted_landscape(particle)
    a_c, b_c = presets.CLASSIC_RATES[particle]
    a_e = el.alpha(V_range)
    b_e = el.beta(V_range)
    xinf_c = a_c(V_range) / (a_c(V_range) + b_c(V_range))
    tau_c = 1.0 / (a_c(V_range) + b_c(V_range))
    xinf_e = el.x_inf(V_range)
    tau_e = el.tau(V_range)

    panels = [
        (axes[0, 0], V_range, a_c(V_range), a_e, "alpha (1/ms)"),
        (axes[0, 1], V_range, b_c(V_range), b_e, "beta (1/ms)"),
        (axes[1, 0], V_range, xinf_c, xinf_e, "x_inf"),
        (axes[1, 1], V_range, tau_c, tau_e, "tau (ms)"),
    ]
    for a, V, c, e, lab in panels:
        a.plot(V, c, color=PALETTE["classic"], lw=1.5, label="classic HH")
        a.plot(V, e, color=PALETTE["eyring"], lw=1.5, ls="--", label="Eyring fit")
        a.set_xlabel("V (mV)")
        a.set_ylabel(lab)
        a.legend(frameon=False, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, savepath)
    return fig, axes


def plot_energy_landscape(el: EnergyLandscape, V_range: np.ndarray, ax=None,
                          savepath: Optional[str] = None,
                          title: Optional[str] = None):
    """Plot the activation barriers (closed/open) vs voltage for an energy landscape."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    if title is None:
        title = f"Energy landscape: {el.name} (z={el.z:.2f}, V_half={el.V_half:.1f} mV)"
    bc = el.barrier_closed(V_range)
    bo = el.barrier_open(V_range)
    ax.plot(V_range, bc, color=PALETTE["Na"], lw=1.5, label="barrier from closed")
    ax.plot(V_range, bo, color=PALETTE["K"], lw=1.5, label="barrier from open")
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xlabel("V (mV)")
    ax.set_ylabel("Activation barrier (kT)")
    ax.set_title(title)
    ax.legend(frameon=False)
    _save(fig, savepath)
    return fig, ax
