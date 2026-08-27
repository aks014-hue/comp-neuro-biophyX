"""Multi-Scale Ion Channel & Hodgkin-Huxley Simulator.

Couples molecular energy-landscape gating -> ion-channel kinetics ->
point-cell electrophysiology for the classic HH channel set.
"""
from .units import thermal_voltage, V_T_6_3
from .energy import EnergyLandscape, fit_to_classic
from . import presets
from .channels import IonChannel, NaChannel, KChannel, LeakChannel
from .cell import PointCell
from .simulator import Simulator, Solution, run, step_pulse
from . import analysis
from . import viz

__all__ = [
    "thermal_voltage", "V_T_6_3",
    "EnergyLandscape", "fit_to_classic",
    "presets",
    "IonChannel", "NaChannel", "KChannel", "LeakChannel",
    "PointCell", "Simulator", "Solution", "run", "step_pulse",
    "analysis", "viz",
]
