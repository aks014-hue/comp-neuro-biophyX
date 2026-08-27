"""Build and execute the 5 analysis notebooks, generating all figures."""
import sys, os, warnings
sys.path.insert(0, "/workspace")
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

FIG_DIR = "/mnt/results/hh_simulator/figures"
NB_DIR = "/mnt/results/hh_simulator/notebooks"
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(NB_DIR, exist_ok=True)

HEADER = (
    "import sys; sys.path.insert(0, '/workspace')\n"
    "import os, warnings; warnings.filterwarnings('ignore')\n"
    "import numpy as np\n"
    "import matplotlib\n"
    "import matplotlib.pyplot as plt\n"
    "matplotlib.rcParams['font.family'] = ['Liberation Sans','Arimo','DejaVu Sans']\n"
    "matplotlib.rcParams['svg.fonttype'] = 'none'\n"
    f"FIG_DIR = '{FIG_DIR}'\n"
    "os.makedirs(FIG_DIR, exist_ok=True)\n"
)

# ============================================================
# Notebook 1: Energy landscape -> rates
# ============================================================
nb1 = new_notebook()
nb1.cells = [
    new_markdown_cell("# 01 — Energy Landscape to Gating Rates\n\n"
        "How molecular energy-landscape parameters (gating charge $z$, half-activation\n"
        "$V_{half}$, symmetry $\\delta$, prefactor $k_0$, rate caps) generate the\n"
        "voltage-dependent transition rates $\\alpha(V)$, $\\beta(V)$, steady-state\n"
        "$x_\\infty(V)$, and time constant $\\tau(V)$ for each HH gating particle.\n\n"
        "A 2-state Eyring model reproduces the Boltzmann $x_\\infty$ well. Rate caps\n"
        "(Kramers limit) prevent $\\tau$ from collapsing at depolarized voltages,\n"
        "matching the saturating behaviour of classic HH rates."),
    new_code_cell(HEADER +
        "from hh_simulator import presets, viz\n"
        "V = np.linspace(-100, 50, 601)\n"
        "fits = {p: presets.fitted_landscape(p) for p in ('m','h','n')}\n"
        "for p in ('m','h','n'): print(fits[p])"),
    new_code_cell(
        "for p in ('m','h','n'):\n"
        "    fig, axes = viz.plot_rates(p, V, savepath=f'{FIG_DIR}/01_rates_{p}.svg')\n"
        "    plt.show()"),
    new_code_cell(
        "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
        "for ax, p in zip(axes, ('m','h','n')):\n"
        "    viz.plot_energy_landscape(fits[p], V, ax=ax)\n"
        "fig.tight_layout()\n"
        "fig.savefig(f'{FIG_DIR}/01_energy_landscapes.svg', bbox_inches='tight')\n"
        "fig.savefig(f'{FIG_DIR}/01_energy_landscapes.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"),
    new_code_cell(
        "print('particle | x_inf err | tau-shape err | tau_max | caps')\n"
        "for p in ('m','h','n'):\n"
        "    a,b = presets.CLASSIC_RATES[p]\n"
        "    xinf_c = a(V)/(a(V)+b(V)); tau_c = 1.0/(a(V)+b(V))\n"
        "    el = fits[p]\n"
        "    ex = np.max(np.abs(el.x_inf(V)-xinf_c))\n"
        "    et = np.max(np.abs(el.tau(V)/np.max(el.tau(V)) - tau_c/np.max(tau_c)))\n"
        "    print(f'{p}        | {ex:.4f}    | {et:.4f}        | {np.max(tau_c):.3f}  | '\n"
        "          f'a={el.alpha_cap:.2f} b={el.beta_cap:.2f}')"),
]
nbf.write(nb1, f"{NB_DIR}/01_energy_landscape_to_rates.ipynb")

# ============================================================
# Notebook 2: Deterministic HH
# ============================================================
nb2 = new_notebook()
nb2.cells = [
    new_markdown_cell("# 02 — Deterministic Hodgkin-Huxley\n\n"
        "Resting state, step-current action potential, gating dynamics, and\n"
        "per-channel ionic currents. Both classic and energy-landscape modes."),
    new_code_cell(HEADER +
        "from hh_simulator import (NaChannel, KChannel, LeakChannel, PointCell,\n"
        "    Simulator, step_pulse, analysis, viz)\n"
        "t_eval = np.linspace(0, 40, 4001)\n"
        "I_fn = step_pulse((0,40), 10.0, onset=5, dur=30)"),
    new_code_cell(
        "cell = PointCell([NaChannel('classic'), KChannel('classic'), LeakChannel()])\n"
        "print(f'Resting potential: {cell.resting_potential():.3f} mV')\n"
        "sim = Simulator(cell)\n"
        "sol = sim.run((0,40), I_inj=I_fn, mode='deterministic', t_eval=t_eval)\n"
        "aps = analysis.detect_aps(sol.V, sol.t)\n"
        "print(f'AP peak={np.max(sol.V):.2f} mV, n_spikes={aps[\"n_spikes\"]}')\n"
        "viz.plot_voltage(sol, savepath=f'{FIG_DIR}/02_voltage_classic.svg')\n"
        "plt.show()"),
    new_code_cell(
        "viz.plot_gating(sol, savepath=f'{FIG_DIR}/02_gating_classic.svg')\n"
        "plt.show()"),
    new_code_cell(
        "viz.plot_currents(sol, savepath=f'{FIG_DIR}/02_currents_classic.svg')\n"
        "plt.show()"),
    new_code_cell(
        "cell_e = PointCell([NaChannel('energy'), KChannel('energy'), LeakChannel()])\n"
        "sol_e = Simulator(cell_e).run((0,40), I_inj=I_fn, mode='deterministic', t_eval=t_eval)\n"
        "fig, ax = plt.subplots(figsize=(7,3.2))\n"
        "ax.plot(sol.t, sol.V, 'k-', lw=1.2, label='classic HH')\n"
        "ax.plot(sol_e.t, sol_e.V, '#E9ED4C', lw=1.2, ls='--', label='Eyring energy')\n"
        "ax.set_xlabel('Time (ms)'); ax.set_ylabel('Voltage (mV)')\n"
        "ax.set_title('Classic vs energy-landscape mode'); ax.legend(frameon=False)\n"
        "fig.savefig(f'{FIG_DIR}/02_classic_vs_energy.svg', bbox_inches='tight')\n"
        "fig.savefig(f'{FIG_DIR}/02_classic_vs_energy.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"),
]
nbf.write(nb2, f"{NB_DIR}/02_deterministic_HH.ipynb")

# ============================================================
# Notebook 3: Stochastic HH
# ============================================================
nb3 = new_notebook()
nb3.cells = [
    new_markdown_cell("# 03 — Stochastic Hodgkin-Huxley\n\n"
        "Channel noise from stochastic single-particle gating (binomial tau-leaping).\n"
        "Effect of channel count on noise and convergence to the deterministic limit."),
    new_code_cell(HEADER +
        "from hh_simulator import (NaChannel, KChannel, LeakChannel, PointCell,\n"
        "    Simulator, step_pulse, analysis, viz)\n"
        "cell = PointCell([NaChannel('classic'), KChannel('classic'), LeakChannel()])\n"
        "sim = Simulator(cell)\n"
        "t_eval = np.linspace(0, 40, 4001)\n"
        "I_fn = step_pulse((0,40), 10.0, onset=5, dur=30)\n"
        "sol_det = sim.run((0,40), I_inj=I_fn, mode='deterministic', t_eval=t_eval)"),
    new_code_cell(
        "rng = np.random.default_rng(42)\n"
        "sol_st = sim.run((0,40), I_inj=I_fn, mode='stochastic', dt=0.01,\n"
        "                 N_channels={'Na':1000,'K':300}, rng=rng, record_every=5)\n"
        "fig, ax = plt.subplots(figsize=(7,3.2))\n"
        "ax.plot(sol_det.t, sol_det.V, 'k-', lw=1, label='deterministic')\n"
        "ax.plot(sol_st.t, sol_st.V, '#0279EE', lw=0.8, alpha=0.7, label='stochastic (N=1000)')\n"
        "ax.set_xlabel('Time (ms)'); ax.set_ylabel('Voltage (mV)')\n"
        "ax.set_title('Channel noise: deterministic vs stochastic'); ax.legend(frameon=False)\n"
        "fig.savefig(f'{FIG_DIR}/03_stochastic_vs_det.svg', bbox_inches='tight')\n"
        "fig.savefig(f'{FIG_DIR}/03_stochastic_vs_det.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"),
    new_code_cell(
        "fig, axes = plt.subplots(3, 1, figsize=(7,6), sharex=True)\n"
        "for ax, p, lab in zip(axes, ('m','h','n'),\n"
        "    ('m (Na activation)','h (Na inactivation)','n (K activation)')):\n"
        "    ax.plot(sol_st.t, sol_st.gating[p], '#0279EE', lw=0.8)\n"
        "    ax.plot(sol_det.t, sol_det.gating[p], 'k--', lw=1, label='deterministic')\n"
        "    ax.set_ylabel(lab); ax.legend(frameon=False, fontsize=8)\n"
        "axes[-1].set_xlabel('Time (ms)')\n"
        "fig.suptitle('Stochastic gating fractions (N_Na=1000)')\n"
        "fig.tight_layout()\n"
        "fig.savefig(f'{FIG_DIR}/03_gating_stochastic.svg', bbox_inches='tight')\n"
        "fig.savefig(f'{FIG_DIR}/03_gating_stochastic.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"),
    new_code_cell(
        "Ns = [100, 300, 1000, 3000, 10000, 30000]\n"
        "variances = []\n"
        "for N in Ns:\n"
        "    rng = np.random.default_rng(123)\n"
        "    s = sim.run((0,60), I_inj=0.0, mode='stochastic', dt=0.01,\n"
        "                N_channels={'Na':N,'K':N//3}, rng=rng, record_every=5)\n"
        "    m = s.t > 20\n"
        "    variances.append(np.var(s.V[m]))\n"
        "fig, ax = plt.subplots(figsize=(5,3.5))\n"
        "ax.loglog(Ns, variances, 'o-', color='#0279EE', lw=1.5)\n"
        "ax.loglog(Ns, [variances[0]*(Ns[0]/n) for n in Ns], 'k--', lw=1, label='$\\\\propto 1/N$')\n"
        "ax.set_xlabel('N (Na channels)'); ax.set_ylabel('Var(V) at rest (mV$^2$)')\n"
        "ax.set_title('Channel-noise variance scaling'); ax.legend(frameon=False)\n"
        "fig.savefig(f'{FIG_DIR}/03_noise_scaling.svg', bbox_inches='tight')\n"
        "fig.savefig(f'{FIG_DIR}/03_noise_scaling.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"),
]
nbf.write(nb3, f"{NB_DIR}/03_stochastic_HH.ipynb")

# ============================================================
# Notebook 4: F-I curve
# ============================================================
nb4 = new_notebook()
nb4.cells = [
    new_markdown_cell("# 04 — F-I Curve\n\n"
        "Firing rate vs injected current. Classic HH exhibits class-2 excitability\n"
        "(discontinuous firing onset). Deterministic and stochastic curves."),
    new_code_cell(HEADER +
        "from hh_simulator import (NaChannel, KChannel, LeakChannel, PointCell,\n"
        "    Simulator, analysis, viz)\n"
        "cell = PointCell([NaChannel('classic'), KChannel('classic'), LeakChannel()])\n"
        "sim = Simulator(cell)\n"
        "I_range = np.arange(0, 25, 1.0)"),
    new_code_cell(
        "I_det, rate_det = analysis.fi_curve(sim, I_range, t_span=(0,300),\n"
        "    onset=20, steady_window=(100,300))\n"
        "viz.plot_fi_curve(I_det, rate_det, savepath=f'{FIG_DIR}/04_fi_deterministic.svg')\n"
        "plt.show()"),
    new_code_cell(
        "I_stoch = np.arange(0, 25, 2.0)\n"
        "trials = []\n"
        "for seed in range(4):\n"
        "    rng = np.random.default_rng(seed)\n"
        "    I_s, rate_s = analysis.fi_curve(sim, I_stoch, t_span=(0,300),\n"
        "        onset=20, steady_window=(100,300), mode='stochastic',\n"
        "        dt=0.01, N_channels={'Na':5000,'K':1500}, rng=rng)\n"
        "    trials.append(rate_s)\n"
        "trials = np.array(trials)\n"
        "fig, ax = plt.subplots(figsize=(5,3.5))\n"
        "ax.errorbar(I_stoch, trials.mean(0), yerr=trials.std(0), fmt='o-',\n"
        "            color='#FF9400', lw=1.5, capsize=3, label='stochastic (N=5000)')\n"
        "ax.plot(I_det, rate_det, 'k-', lw=1, alpha=0.5, label='deterministic')\n"
        "ax.set_xlabel('Injected current (uA/cm^2)'); ax.set_ylabel('Firing rate (Hz)')\n"
        "ax.set_title('F-I curve: deterministic vs stochastic'); ax.legend(frameon=False)\n"
        "fig.savefig(f'{FIG_DIR}/04_fi_comparison.svg', bbox_inches='tight')\n"
        "fig.savefig(f'{FIG_DIR}/04_fi_comparison.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"),
]
nbf.write(nb4, f"{NB_DIR}/04_FI_curve.ipynb")

# ============================================================
# Notebook 5: Validation
# ============================================================
nb5 = new_notebook()
nb5.cells = [
    new_markdown_cell("# 05 — Validation\n\n"
        "Benchmarks: rate matching, action-potential comparison, resting potential,\n"
        "stochastic convergence, and singularity handling."),
    new_code_cell(HEADER +
        "from hh_simulator import (presets, NaChannel, KChannel, LeakChannel,\n"
        "    PointCell, Simulator, step_pulse, analysis, viz)\n"
        "V = np.linspace(-100, 50, 601)"),
    new_code_cell(
        "fig, axes = plt.subplots(1, 2, figsize=(10, 4))\n"
        "for p, c in zip(('m','h','n'), ('#0279EE','#FD9BED','#FF9400')):\n"
        "    a,b = presets.CLASSIC_RATES[p]\n"
        "    el = presets.fitted_landscape(p)\n"
        "    xinf_c = a(V)/(a(V)+b(V)); tau_c = 1.0/(a(V)+b(V))\n"
        "    axes[0].plot(V, el.x_inf(V), color=c, lw=1.5, label=f'{p} Eyring')\n"
        "    axes[0].plot(V, xinf_c, color=c, lw=1, ls=':', label=f'{p} classic')\n"
        "    axes[1].plot(V, el.tau(V), color=c, lw=1.5)\n"
        "    axes[1].plot(V, tau_c, color=c, lw=1, ls=':')\n"
        "axes[0].set_xlabel('V (mV)'); axes[0].set_ylabel('x_inf')\n"
        "axes[0].legend(frameon=False, fontsize=8)\n"
        "axes[1].set_xlabel('V (mV)'); axes[1].set_ylabel('tau (ms)')\n"
        "fig.suptitle('Rate matching: Eyring (solid) vs classic HH (dotted)')\n"
        "fig.tight_layout()\n"
        "fig.savefig(f'{FIG_DIR}/05_rate_matching.svg', bbox_inches='tight')\n"
        "fig.savefig(f'{FIG_DIR}/05_rate_matching.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"),
    new_code_cell(
        "t_eval = np.linspace(0, 40, 4001)\n"
        "I_fn = step_pulse((0,40), 10.0, onset=5, dur=30)\n"
        "cell_c = PointCell([NaChannel('classic'), KChannel('classic'), LeakChannel()])\n"
        "cell_e = PointCell([NaChannel('energy'), KChannel('energy'), LeakChannel()])\n"
        "sol_c = Simulator(cell_c).run((0,40), I_inj=I_fn, mode='deterministic', t_eval=t_eval)\n"
        "sol_e = Simulator(cell_e).run((0,40), I_inj=I_fn, mode='deterministic', t_eval=t_eval)\n"
        "fig, ax = plt.subplots(figsize=(7,3.2))\n"
        "ax.plot(t_eval, sol_c.V, 'k-', lw=1.2, label='classic')\n"
        "ax.plot(t_eval, sol_e.V, '#E9ED4C', lw=1.2, ls='--', label='energy')\n"
        "ax.set_xlabel('Time (ms)'); ax.set_ylabel('Voltage (mV)')\n"
        "ax.set_title(f'AP: classic peak={np.max(sol_c.V):.1f} mV, energy peak={np.max(sol_e.V):.1f} mV')\n"
        "ax.legend(frameon=False)\n"
        "fig.savefig(f'{FIG_DIR}/05_ap_comparison.svg', bbox_inches='tight')\n"
        "fig.savefig(f'{FIG_DIR}/05_ap_comparison.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"),
    new_code_cell(
        "rmsds = []; Ns = [1000, 3000, 10000, 30000, 100000]\n"
        "for N in Ns:\n"
        "    rng = np.random.default_rng(7)\n"
        "    s = Simulator(cell_c).run((0,40), I_inj=I_fn, mode='stochastic', dt=0.01,\n"
        "        N_channels={'Na':N,'K':N//3}, rng=rng, record_every=5)\n"
        "    Vi = np.interp(t_eval, s.t, s.V)\n"
        "    sub = sol_c.V < -30\n"
        "    rmsds.append(np.sqrt(np.mean((Vi[sub]-sol_c.V[sub])**2)))\n"
        "fig, ax = plt.subplots(figsize=(5,3.5))\n"
        "ax.loglog(Ns, rmsds, 'o-', color='#0279EE', lw=1.5)\n"
        "ax.set_xlabel('N (Na channels)'); ax.set_ylabel('Subthreshold RMSD (mV)')\n"
        "ax.set_title('Stochastic convergence to deterministic')\n"
        "fig.savefig(f'{FIG_DIR}/05_convergence.svg', bbox_inches='tight')\n"
        "fig.savefig(f'{FIG_DIR}/05_convergence.png', bbox_inches='tight', dpi=150)\n"
        "plt.show()"),
    new_code_cell(
        "print('=== Validation Summary ===')\n"
        "print(f'Resting potential (classic): {cell_c.resting_potential():.3f} mV')\n"
        "print(f'Resting potential (energy):  {cell_e.resting_potential():.3f} mV')\n"
        "print(f'AP peak (classic): {np.max(sol_c.V):.2f} mV')\n"
        "print(f'AP peak (energy):  {np.max(sol_e.V):.2f} mV')\n"
        "print(f'alpha_n(-55) = {float(presets.alpha_n(-55.0)):.4f} (singularity OK)')\n"
        "print(f'alpha_m(-40) = {float(presets.alpha_m(-40.0)):.4f} (singularity OK)')\n"
        "for p in ('m','h','n'):\n"
        "    a,b = presets.CLASSIC_RATES[p]\n"
        "    xinf_c = a(V)/(a(V)+b(V))\n"
        "    el = presets.fitted_landscape(p)\n"
        "    print(f'{p}: x_inf max err = {np.max(np.abs(el.x_inf(V)-xinf_c)):.4f}')\n"),
]
nbf.write(nb5, f"{NB_DIR}/05_validation.ipynb")

print(f"Created 5 notebooks in {NB_DIR}")
