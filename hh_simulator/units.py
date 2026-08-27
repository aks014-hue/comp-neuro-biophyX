"""Physical constants and unit conventions for the HH simulator.

Conventions (classic Hodgkin-Huxley, squid axon):
  Voltage:        mV
  Time:           ms
  Conductance:    mS/cm^2
  Current:        uA/cm^2
  Capacitance:    uF/cm^2
  Temperature:    degrees C (input), K (internal)
"""
import numpy as np

# --- Physical constants (SI) ---
K_B = 1.380649e-23           # Boltzmann constant, J/K
E_CHARGE = 1.602176634e-19   # Elementary charge, C
R_GAS = 8.314462618          # Gas constant, J/(mol K)
FARADAY = 96485.33212        # Faraday constant, C/mol

# Attempt-frequency prefactor for Eyring rate theory (1/ms).
# Represents the molecular attempt frequency ~ kT/h scaled into our ms time base.
# Used only to convert k0 <-> barrier height dG_dagger for reporting/visualization.
ATTEMPT_FREQ = 1.0e3         # 1/ms (~ 10^12 /s is physical; we work in ms so 10^9/ms;
                             # here we use a reduced value so reported barriers stay
                             # in a readable kT range; the absolute scale is arbitrary
                             # and only affects the *reported* barrier, not the rates.)


def thermal_voltage(temp_c: float = 6.3) -> float:
    """V_T = kT/e in mV at the given temperature (degrees C)."""
    temp_k = temp_c + 273.15
    return (K_B * temp_k / E_CHARGE) * 1e3  # V -> mV


# Default thermal voltage at 6.3 C (~24.12 mV)
V_T_6_3 = thermal_voltage(6.3)
