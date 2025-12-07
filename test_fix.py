#!/usr/bin/env python
"""Quick test of the find_Fmin fix"""

import numpy as np
from scipy.special import roots_legendre
from income_distribution import find_Fmin

# Set up quadrature
xi, wi = roots_legendre(32)

# Test parameters (from the error message)
y_mean_before_damage = 95218.53164757002
omega_base = 0.47051037779438315
y_damage_distribution_exponent = 0.5
y_net_reference = 17210.79
uniform_redistribution = 0.0
gini = 0.5
target_subsidy = 1.7138956544975259e-09

print(f"Testing find_Fmin with target_subsidy={target_subsidy}")
print(f"y_mean_before_damage={y_mean_before_damage}")

try:
    Fmin_result = find_Fmin(
        y_mean_before_damage,
        omega_base,
        y_damage_distribution_exponent,
        y_net_reference,
        uniform_redistribution,
        gini,
        xi,
        wi,
        target_subsidy=target_subsidy
    )
    print(f"SUCCESS: Fmin = {Fmin_result}")
except Exception as e:
    print(f"FAILED: {e}")
