# Computational Biophysics & Neuroscience Simulator

Numerical simulation suite for neuronal membrane dynamics, Hodgkin-Huxley kinetics, and electrophysiological ion transport.

## Repository Structure

* **`hh_simulator/`**: Python module implementing RK4 integration for voltage-gated Na+ and K+ channels.
* **`notebooks/`**: Execution workflows covering action potential dynamics, F-I firing curves, and numerical stability tests.
* **`tests/`**: Unit tests verifying kinetic rate constants and membrane capacitance balance.
* **`figures/`**: Rendered voltage-clamp trajectories, phase-plane diagrams, and ion channel open probabilities.

## Quickstart

```bash
git clone [https://github.com/aks014-hue/comp-neuro-biophysics.git](https://github.com/aks014-hue/comp-neuro-biophysics.git)
cd comp-neuro-biophysics
python build_notebooks.py
