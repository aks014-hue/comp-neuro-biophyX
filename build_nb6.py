"""Build notebook 06 (tau error localization) without touching notebooks 1-5."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

FIG_DIR = "/mnt/results/hh_simulator/figures"
NB_DIR = "/mnt/results/hh_simulator/notebooks"

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

ANALYSIS = r"""
V = np.linspace(-100, 50, 2001)
LANDMARKS = [(-65.0, 'rest'), (-55.0, 'thresh'), (40.0, 'peak')]
colors = {'m': '#0279EE', 'h': '#FD9BED', 'n': '#FF9400'}
labels = {'m': 'm (Na act.)', 'h': 'h (Na inact.)', 'n': 'n (K act.)'}

# Reference AP trajectory (classic mode) for trajectory-weighted error
cell = PointCell([NaChannel('classic'), KChannel('classic'), LeakChannel()])
sol = Simulator(cell).run((0, 40), I_inj=step_pulse((0, 40), 10.0, onset=5, dur=30),
                          mode='deterministic', t_eval=np.linspace(0, 40, 4001))

fig, axes = plt.subplots(3, 2, figsize=(11, 8.5), sharex=True)
rows = {}
for i, p in enumerate(('m', 'h', 'n')):
    a_c, b_c = presets.CLASSIC_RATES[p]
    tau_c = 1.0 / (a_c(V) + b_c(V))
    el = presets.fitted_landscape(p)
    tau_e = el.tau(V)
    rel = (tau_e - tau_c) / tau_c * 100.0

    # cap-active region (alpha()/beta() return capped rates)
    a_e, b_e = el.alpha(V), el.beta(V)
    cap_mask = np.zeros(V.shape, dtype=bool)
    if np.isfinite(el.alpha_cap):
        cap_mask |= a_e >= 0.999 * el.alpha_cap
    if np.isfinite(el.beta_cap):
        cap_mask |= b_e >= 0.999 * el.beta_cap
    v_cap = V[np.argmax(cap_mask)] if cap_mask.any() else np.nan

    axL, axR = axes[i]
    axL.semilogy(V, tau_c, 'k-', lw=1.5, label='classic HH')
    axL.semilogy(V, tau_e, color=colors[p], ls='--', lw=1.5, label='Eyring + cap')
    if np.isfinite(v_cap):
        for ax in (axL, axR):
            ax.axvspan(v_cap, V[-1], color=colors[p], alpha=0.10, lw=0)
    axL.set_ylabel(labels[p] + '\n' + r'$\tau$ (ms)')
    axL.legend(frameon=True, framealpha=0.95, edgecolor='none', fontsize=8, loc='lower left')

    axR.plot(V, rel, color=colors[p], lw=1.5)
    axR.axhline(0, color='k', lw=0.6)
    for vv, lab in LANDMARKS:
        axR.axvline(vv, color='gray', ls=':', lw=0.8)
    j = int(np.argmax(np.abs(rel)))
    axR.plot(V[j], rel[j], 'o', color='k', ms=4)
    axR.set_ylabel(r'$\tau$ error (%)')
    if i == 0:
        for vv, lab in LANDMARKS:
            axR.annotate(lab, (vv, axR.get_ylim()[1]), textcoords='offset points',
                         xytext=(3, -2), fontsize=7, color='gray', va='top')

    at = lambda vv: float(np.interp(vv, V, rel))
    sub = V < -55.0
    rel_ap = np.interp(sol.V, V, rel)          # error along the AP trajectory
    rows[p] = dict(
        v_cap=v_cap, vmax=V[j], emax=rel[j],
        e_rest=at(-65.0), e_thresh=at(-55.0), e_peak=at(40.0),
        m_sub=float(np.mean(np.abs(rel[sub]))),
        m_supra=float(np.mean(np.abs(rel[~sub]))),
        m_ap=float(np.mean(np.abs(rel_ap))),
        tau_c_rest=float(np.interp(-65.0, V, tau_c)),
    )

axes[2, 0].set_xlabel('V (mV)'); axes[2, 1].set_xlabel('V (mV)')
axes[0, 0].set_title('Time constants'); axes[0, 1].set_title('Signed relative error')
fig.suptitle('Tau error localization')
fig.tight_layout()
fig.savefig(f'{FIG_DIR}/06_tau_error_localization.svg', bbox_inches='tight')
fig.savefig(f'{FIG_DIR}/06_tau_error_localization.png', bbox_inches='tight', dpi=150)
plt.show()

print(f"{'p':<2} {'cap_on':>7} {'V@maxE':>7} {'maxE%':>7} {'rest%':>7} {'thr%':>7} {'peak%':>7} "
      f"{'m|sub|':>7} {'m|sup|':>7} {'m|AP|':>7} {'tau_c(rest)':>11}")
for p, r in rows.items():
    print(f"{p:<2} {r['v_cap']:>7.1f} {r['vmax']:>7.1f} {r['emax']:>7.1f} {r['e_rest']:>7.1f} "
          f"{r['e_thresh']:>7.1f} {r['e_peak']:>7.1f} {r['m_sub']:>7.1f} {r['m_supra']:>7.1f} "
          f"{r['m_ap']:>7.2f} {r['tau_c_rest']:>11.3f}")
print('\nFraction of 40 ms AP trace spent with V > cap onset:')
for p, r in rows.items():
    frac = float(np.mean(sol.V > r['v_cap']))
    print(f'  {p}: V > {r["v_cap"]:.1f} mV for {frac*100:.1f}% of the trace')
"""

nb = new_notebook()
nb.cells = [
    new_markdown_cell(
        "# 06 — Tau Error Localization\n\n"
        "Where in voltage space does the residual tau mismatch between the capped\n"
        "Eyring model and classic HH live, and is the rate cap surgical or global?\n\n"
        "Left: tau(V) for classic HH (black) and Eyring+cap (colored dashed); the\n"
        "shaded band marks voltages where the rate cap is engaged. Right: signed\n"
        "relative tau error, with rest / threshold / AP-peak landmarks and the peak\n"
        "error marked. The summary table adds an AP-trajectory-weighted mean error\n"
        "(the error the membrane actually experiences during a spike)."),
    new_code_cell(HEADER +
        "from hh_simulator import (presets, NaChannel, KChannel, LeakChannel,\n"
        "    PointCell, Simulator, step_pulse)"),
    new_code_cell(ANALYSIS),
]
nbf.write(nb, f"{NB_DIR}/06_tau_error_localization.ipynb")
print(f"Wrote {NB_DIR}/06_tau_error_localization.ipynb")
