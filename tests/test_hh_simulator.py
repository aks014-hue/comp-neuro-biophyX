"""Test suite for the hh_simulator package.

Run with: pytest /workspace/tests/ -v
"""
import sys
sys.path.insert(0, "/workspace")

import numpy as np
import pytest

from hh_simulator import (presets, EnergyLandscape, fit_to_classic,
    NaChannel, KChannel, LeakChannel, PointCell, Simulator, step_pulse,
    analysis, viz)
from hh_simulator.units import thermal_voltage, V_T_6_3


# ------------------------------------------------------------------
# Energy landscape
# ------------------------------------------------------------------
class TestEnergyLandscape:
    def test_rates_positive(self):
        el = presets.fitted_landscape("m")
        V = np.linspace(-100, 50, 101)
        assert np.all(el.alpha(V) > 0)
        assert np.all(el.beta(V) > 0)

    def test_x_inf_bounded(self):
        el = presets.fitted_landscape("h")
        V = np.linspace(-100, 50, 101)
        xinf = el.x_inf(V)
        assert np.all(xinf >= 0) and np.all(xinf <= 1)

    def test_tau_positive(self):
        el = presets.fitted_landscape("n")
        V = np.linspace(-100, 50, 101)
        assert np.all(el.tau(V) > 0)

    def test_x_inf_midpoint(self):
        el = presets.fitted_landscape("m")
        assert abs(el.x_inf(el.V_half) - 0.5) < 1e-6

    def test_fit_matches_classic_x_inf(self):
        V = np.linspace(-100, 50, 601)
        for p in ("m", "h", "n"):
            a, b = presets.CLASSIC_RATES[p]
            xinf_c = a(V) / (a(V) + b(V))
            el = presets.fitted_landscape(p)
            err = np.max(np.abs(el.x_inf(V) - xinf_c))
            assert err < 0.08, f"{p}: x_inf error {err:.4f} exceeds 0.08"

    def test_rate_cap_prevents_tau_collapse(self):
        """With caps, tau at depolarized V should not collapse to ~0."""
        el = presets.fitted_landscape("h")
        tau_at_0 = float(el.tau(0.0))
        assert tau_at_0 > 0.5, f"h tau at V=0 is {tau_at_0:.3f}, should be >0.5 ms"

    def test_activation_vs_inactivation_sign(self):
        assert presets.fitted_landscape("m").z > 0   # activation
        assert presets.fitted_landscape("n").z > 0   # activation
        assert presets.fitted_landscape("h").z < 0   # inactivation


# ------------------------------------------------------------------
# Classic rate singularities
# ------------------------------------------------------------------
class TestClassicRates:
    def test_alpha_n_singularity(self):
        assert np.isfinite(presets.alpha_n(-55.0))
        assert abs(float(presets.alpha_n(-55.0)) - 0.1) < 1e-3

    def test_alpha_m_singularity(self):
        assert np.isfinite(presets.alpha_m(-40.0))
        assert abs(float(presets.alpha_m(-40.0)) - 1.0) < 1e-3

    def test_rates_finite_over_range(self):
        V = np.linspace(-100, 50, 101)
        for fn in [presets.alpha_n, presets.beta_n, presets.alpha_m,
                   presets.beta_m, presets.alpha_h, presets.beta_h]:
            assert np.all(np.isfinite(fn(V)))


# ------------------------------------------------------------------
# Channels
# ------------------------------------------------------------------
class TestChannels:
    def test_na_conductance_factor(self):
        ch = NaChannel("classic")
        state = {"m": 1.0, "h": 1.0}
        assert abs(ch.conductance_factor(state) - 1.0) < 1e-10
        state = {"m": 0.0, "h": 1.0}
        assert abs(ch.conductance_factor(state) - 0.0) < 1e-10

    def test_k_conductance_factor(self):
        ch = KChannel("classic")
        state = {"n": 1.0}
        assert abs(ch.conductance_factor(state) - 1.0) < 1e-10

    def test_leak_conductance(self):
        ch = LeakChannel()
        assert abs(ch.conductance_factor({}) - 1.0) < 1e-10

    def test_na_current_sign(self):
        ch = NaChannel("classic")
        state = {"m": 1.0, "h": 1.0}
        I = ch.current(0.0, state)  # V=0 < E_Na=50 -> inward (negative)
        assert I < 0
        I = ch.current(60.0, state)  # V=60 > E_Na=50 -> outward (positive)
        assert I > 0


# ------------------------------------------------------------------
# Point cell
# ------------------------------------------------------------------
class TestPointCell:
    def test_resting_potential(self):
        cell = PointCell([NaChannel("classic"), KChannel("classic"), LeakChannel()])
        Vr = cell.resting_potential()
        assert abs(Vr - (-65.0)) < 1.0

    def test_energy_mode_resting(self):
        cell = PointCell([NaChannel("energy"), KChannel("energy"), LeakChannel()])
        Vr = cell.resting_potential()
        assert abs(Vr - (-65.0)) < 2.0

    def test_zero_current_at_rest(self):
        cell = PointCell([NaChannel("classic"), KChannel("classic"), LeakChannel()])
        Vr = cell.resting_potential()
        state = cell.steady_state(Vr)
        assert abs(cell.total_ionic_current(Vr, state)) < 1e-6


# ------------------------------------------------------------------
# Simulator: deterministic
# ------------------------------------------------------------------
class TestDeterministic:
    def test_resting_no_current(self):
        cell = PointCell([NaChannel("classic"), KChannel("classic"), LeakChannel()])
        sim = Simulator(cell)
        sol = sim.run((0, 50), I_inj=0.0, mode="deterministic",
                      t_eval=np.linspace(0, 50, 501))
        assert np.max(np.abs(sol.V - sol.V[0])) < 0.1  # stays at rest

    def test_action_potential(self):
        cell = PointCell([NaChannel("classic"), KChannel("classic"), LeakChannel()])
        sim = Simulator(cell)
        I_fn = step_pulse((0, 40), 10.0, onset=5, dur=30)
        sol = sim.run((0, 40), I_inj=I_fn, mode="deterministic",
                      t_eval=np.linspace(0, 40, 4001))
        assert np.max(sol.V) > 30.0  # AP peak > 30 mV
        aps = analysis.detect_aps(sol.V, sol.t)
        assert aps["n_spikes"] >= 1

    def test_energy_mode_ap(self):
        cell = PointCell([NaChannel("energy"), KChannel("energy"), LeakChannel()])
        sim = Simulator(cell)
        I_fn = step_pulse((0, 40), 10.0, onset=5, dur=30)
        sol = sim.run((0, 40), I_inj=I_fn, mode="deterministic",
                      t_eval=np.linspace(0, 40, 4001))
        assert np.max(sol.V) > 30.0  # energy-mode also produces AP


# ------------------------------------------------------------------
# Simulator: stochastic
# ------------------------------------------------------------------
class TestStochastic:
    def test_stochastic_runs(self):
        cell = PointCell([NaChannel("classic"), KChannel("classic"), LeakChannel()])
        sim = Simulator(cell)
        rng = np.random.default_rng(42)
        sol = sim.run((0, 20), I_inj=0.0, mode="stochastic", dt=0.01,
                      N_channels={"Na": 1000, "K": 300}, rng=rng, record_every=10)
        assert len(sol.t) > 0
        assert sol.mode == "stochastic"

    def test_noise_variance_scales_with_N(self):
        """Variance at rest should decrease as channel count increases."""
        cell = PointCell([NaChannel("classic"), KChannel("classic"), LeakChannel()])
        sim = Simulator(cell)
        variances = []
        for N in [100, 1000, 10000]:
            rng = np.random.default_rng(123)
            sol = sim.run((0, 60), I_inj=0.0, mode="stochastic", dt=0.01,
                          N_channels={"Na": N, "K": N // 3}, rng=rng, record_every=5)
            mask = sol.t > 20
            variances.append(np.var(sol.V[mask]))
        # variance should decrease with N (at least N=10000 < N=100)
        assert variances[2] < variances[0]


# ------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------
class TestAnalysis:
    def test_ap_detection_synthetic(self):
        t = np.linspace(0, 100, 10001)
        V = -65.0 + 100.0 * np.exp(-((t - 20) / 1.0) ** 2) + \
            100.0 * np.exp(-((t - 50) / 1.0) ** 2)
        aps = analysis.detect_aps(V, t, threshold=0.0, refractory=5.0)
        assert aps["n_spikes"] == 2
        assert abs(aps["spike_times"][0] - 20.0) < 0.5
        assert abs(aps["spike_times"][1] - 50.0) < 0.5

    def test_fi_curve(self):
        cell = PointCell([NaChannel("classic"), KChannel("classic"), LeakChannel()])
        sim = Simulator(cell)
        I_range = np.array([0.0, 5.0, 10.0, 15.0, 20.0])
        I, rate = analysis.fi_curve(sim, I_range,
                                    t_span=(0, 200), onset=20,
                                    steady_window=(50, 200))
        # zero current -> zero firing
        assert rate[0] == 0.0
        # high current -> nonzero firing
        assert rate[-1] > 0.0
