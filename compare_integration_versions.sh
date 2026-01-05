#!/bin/bash

# Compare version 008 (git HEAD) vs version 009 (current) forward integration accuracy

# Create a simple test config with fixed f and s
cat > json/test_fixed_control.json << 'EOFJSON'
{
    "run_name": "test_fixed_control",
    "description": "Fixed control test for version comparison",
    "scalar_parameters": {
        "alpha": 0.3,
        "delta": 0.1,
        "psi1": 0,
        "psi2": 0.013927,
        "k_climate": 5.6869e-13,
        "eta": 0.95,
        "rho": 0.01,
        "fract_gdp": 1.0,
        "theta2": 2.6,
        "mu_max": 1e+20,
        "Ecum_initial": 2193031000000.0,
        "K_initial": 295000000000000.0,
        "y_net_reference": 17210.79,
        "y_damage_distribution_exponent": 0.41,
        "income_dependent_damage_distribution": false,
        "income_dependent_aggregate_damage": false,
        "income_dependent_tax_policy": true,
        "income_redistribution": true,
        "income_dependent_redistribution_policy": true,
        "use_empirical_lorenz": true
    },
    "time_functions": {
        "A": {"type": "gompertz_growth", "initial_value": 739.619, "final_value": 4906815.811, "adjustment_coefficient": -0.0015},
        "L": {"type": "gompertz_growth", "initial_value": 7752000000.0, "final_value": 10825000000.0, "adjustment_coefficient": -0.03133},
        "sigma": {"type": "gompertz_growth", "initial_value": 0.000291355, "final_value": 4.4681e-05, "adjustment_coefficient": -0.008164},
        "theta1": {"type": "double_exponential_growth", "initial_value": 695.177385, "growth_rate_1": -0.023, "growth_rate_2": -0.00065, "fract_1": 0.310452},
        "gini": {"type": "exponential_growth", "exponential_scaling": 0.19517, "additive_constant": 0.37796, "growth_rate": -0.01058},
        "f": {"type": "constant", "value": 0.02},
        "s": {"type": "constant", "value": 0.28}
    },
    "integration_parameters": {
        "t_start": 0.0,
        "t_end": 400,
        "dt": 1.0,
        "n_quad": 32,
        "plot_short_horizon": 100.0
    }
}
EOFJSON

echo "========================================================================"
echo "Integration Accuracy Test: Version 008 (git HEAD) vs Version 009 (current)"
echo "========================================================================"
echo ""
echo "Using FIXED control: f=0.02, s=0.28"
echo "Flags: t-t-t (income-dependent tax + redistribution + targeted)"
echo ""

# Save current changes
echo "Step 1: Saving current version..."
git stash push -m "Version 009 changes"

# Run with version 008
echo "Step 2: Running with version 008 (git HEAD)..."
python test_integration.py json/test_fixed_control.json > /tmp/test_008_output.txt 2>&1
mv data/output/test_fixed_control_* /tmp/test_fixed_control_008 2>/dev/null

# Restore version 009
echo "Step 3: Restoring version 009..."
git stash pop

# Run with version 009
echo "Step 4: Running with version 009 (current)..."
python test_integration.py json/test_fixed_control.json > /tmp/test_009_output.txt 2>&1
mv data/output/test_fixed_control_* /tmp/test_fixed_control_009 2>/dev/null

# Compare results
echo "Step 5: Comparing results..."
python3 << 'EOFPYTHON'
import numpy as np
import csv
from pathlib import Path

def load_csv(dir_path):
    csv_files = list(dir_path.glob("*.csv"))
    if not csv_files:
        return None
    
    data = {}
    with open(csv_files[0]) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key in row.keys():
                clean_key = key.split(',')[0].strip()
                if clean_key not in data:
                    data[clean_key] = []
                try:
                    data[clean_key].append(float(row[key]))
                except:
                    pass
    
    return {k: np.array(v) for k, v in data.items() if v}

dir_008 = Path("/tmp/test_fixed_control_008")
dir_009 = Path("/tmp/test_fixed_control_009")

data_008 = load_csv(dir_008)
data_009 = load_csv(dir_009)

if data_008 and data_009:
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON (Fixed control: f=0.02, s=0.28, t-t-t flags)")
    print("=" * 80)
    
    vars_to_check = ['Omega', 'delta_T', 'Ecum', 'Y_net', 'U', 'K', 'Omega_base']
    
    for var in vars_to_check:
        if var in data_008 and var in data_009:
            v008 = data_008[var]
            v009 = data_009[var]
            
            mean_rel_diff = np.mean(np.abs(v009 - v008) / np.abs(v008)) * 100
            max_rel_diff = np.max(np.abs(v009 - v008) / np.abs(v008)) * 100
            
            print(f"\n{var}:")
            print(f"  Mean (008):        {np.mean(v008):15.8e}")
            print(f"  Mean (009):        {np.mean(v009):15.8e}")
            print(f"  Mean rel diff:     {mean_rel_diff:8.4f}%")
            print(f"  Max rel diff:      {max_rel_diff:8.4f}%")
            
            if max_rel_diff > 0.01:
                print(f"  → SIGNIFICANT DIFFERENCE")
            else:
                print(f"  → Good agreement")
    
    print("\n" + "=" * 80)
    print("Interpretation:")
    print("  If Omega_base differs: Climate trajectory diverged")
    print("  If Omega differs but Omega_base matches: Damage distribution issue")
    print("  Small differences (<0.01%): Numerical noise only")
    print("=" * 80)
else:
    print("ERROR: Could not load results")
EOFPYTHON

