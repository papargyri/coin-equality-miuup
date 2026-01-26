#!/usr/bin/env python3
"""
Create Excel workbook comparing DICE and forward model at 2020 only.

All variables in a single sheet as rows for easy comparison.

Usage:
    python create_bn_comparison_2020_xlsx.py [--output <path>]
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path


def load_bn_results(scenario='optimal'):
    """Load B&N results from JSON file."""
    path = Path(f'barrage_nordhaus_2023/bn_results_{scenario}.json')
    with open(path, 'r') as f:
        return json.load(f)


def load_bn_raw_results():
    """Load raw B&N results (original units)."""
    path = Path('barrage_nordhaus_2023/bn_results_raw.json')
    with open(path, 'r') as f:
        return json.load(f)


def load_forward_results():
    """Load our forward simulation results."""
    path = Path('data/output/bn_comparison/our_results_optimal.json')
    with open(path, 'r') as f:
        return json.load(f)


def get_value_at_2020(data, var_name, time_key='year'):
    """Get value at 2020 from a time series."""
    if var_name not in data:
        return np.nan

    times = np.array(data[time_key])
    values = np.array(data[var_name])

    # Find index closest to 2020
    idx = np.argmin(np.abs(times - 2020))
    return values[idx]


def main():
    parser = argparse.ArgumentParser(
        description='Create B&N comparison Excel workbook for 2020 only'
    )
    parser.add_argument('--output', type=str,
                       default='data/output/bn_comparison/dice_vs_forward_2020.xlsx',
                       help='Output xlsx path')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    # Load B&N results (converted units)
    bn_data = load_bn_results('optimal')

    # Load B&N raw results (original units)
    bn_raw = load_bn_raw_results()

    # Load forward results
    forward_results = load_forward_results()

    print(f"Creating Excel workbook: {output_path}")

    # Define all variables to compare
    variables = [
        # (Variable name, DICE var, Our var, Units, Description)
        ('Emission Control Rate', 'MIU', 'mu', '-', 'μ: Fraction of emissions abated'),
        ('Savings Rate', 'S', 's', '-', 's: Fraction of output saved/invested'),
        ('Capital Stock', 'K', 'K', '$', 'K: Total capital'),
        ('Gross Output', 'Y_gross', 'Y_gross', '$', 'Y_gross: Output before damages and abatement'),
        ('Net Output', 'Y_net', 'Y_net', '$', 'Y_net: Output after damages and abatement'),
        ('Cumulative Emissions', 'Ecum', 'Ecum', 'tCO2', 'Ecum: Cumulative CO2 since pre-industrial'),
        ('Annual Emissions', 'E', 'E', 'tCO2/yr', 'E: Annual CO2 emissions'),
        ('Temperature Change', 'delta_T', 'delta_T', '°C', 'ΔT: Temperature above pre-industrial'),
        ('Damage Fraction', 'Omega', 'Omega', '-', 'Ω: Climate damages as fraction of output'),
        ('Consumption per Capita', 'consumption', 'consumption', '$/person', 'c: Per-capita consumption'),
        ('Population', 'L', 'L', 'persons', 'L: World population'),
        ('TFP', 'A', 'A', '-', 'A: Total factor productivity (our units)'),
        ('Carbon Intensity', 'sigma', 'sigma', 'tCO2/$', 'σ: CO2 per dollar of output'),
        ('Backstop Price', 'theta1', 'theta1', '$/tCO2', 'θ₁: Cost to abate last ton of CO2'),
    ]

    # Build the comparison data
    rows = []
    for var_name, bn_var, our_var, units, description in variables:
        dice_val = get_value_at_2020(bn_data, bn_var)
        our_val = get_value_at_2020(forward_results, our_var, time_key='t')

        # Compute difference
        if np.isnan(dice_val) or np.isnan(our_val):
            diff = np.nan
            diff_pct = np.nan
        else:
            diff = our_val - dice_val
            diff_pct = 100 * diff / dice_val if dice_val != 0 else (0 if our_val == 0 else np.inf)

        rows.append({
            'Variable': var_name,
            'Symbol': description.split(':')[0] if ':' in description else '',
            'Units': units,
            'DICE (B&N 2023)': dice_val,
            'Our Forward Model': our_val,
            'Difference': diff,
            'Difference (%)': diff_pct,
            'Description': description,
        })

    # Create DataFrame
    df_comparison = pd.DataFrame(rows)

    # Also create a parameters comparison
    params = [
        ('Capital Share', 'α', '-', 0.3, 0.3, 'Output elasticity of capital'),
        ('Depreciation Rate', 'δ', '1/yr', 0.1, 0.1, 'Annual capital depreciation'),
        ('Linear Damage Coef', 'ψ₁', '1/°C', 0.0, 0.0, 'Linear term in damage function'),
        ('Quadratic Damage Coef', 'ψ₂', '1/°C²', 0.003467, 0.003467, 'Quadratic term in damage function'),
        ('Abatement Exponent', 'θ₂', '-', 2.6, 2.6, 'Exponent in abatement cost function'),
        ('Elasticity of MU', 'η', '-', 0.95, 0.95, 'Elasticity of marginal utility'),
        ('Time Preference', 'ρ', '1/yr', 0.026341, 0.026341, 'Pure rate of time preference'),
        ('Climate Sensitivity', 'k_climate', '°C/tCO2', 'FAIR model', 5.6869e-13, 'Temperature per unit cumulative emissions'),
        ('Initial Capital', 'K₀', 'trillion $', 295, 295, 'Capital stock at 2020'),
        ('Initial Cum. Emissions', 'Ecum₀', 'tCO2', 2.321e12, 2.193e12, 'Cumulative emissions at 2020 (calibrated for T)'),
        ('Initial Temperature', 'T₀', '°C', 1.247, 1.247, 'Temperature at 2020'),
    ]

    param_rows = []
    for param_name, symbol, units, dice_val, our_val, description in params:
        if isinstance(dice_val, str) or isinstance(our_val, str):
            diff = 'N/A'
            diff_pct = 'N/A'
        else:
            diff = our_val - dice_val
            diff_pct = 100 * diff / dice_val if dice_val != 0 else 0
            diff_pct = f'{diff_pct:.2f}%'

        param_rows.append({
            'Parameter': param_name,
            'Symbol': symbol,
            'Units': units,
            'DICE (B&N 2023)': dice_val,
            'Our Forward Model': our_val,
            'Difference (%)': diff_pct,
            'Description': description,
        })

    df_params = pd.DataFrame(param_rows)

    # Add raw DICE values for reference
    raw_vars = [
        ('MIU', 'Emission Control Rate', '-'),
        ('S', 'Savings Rate', '-'),
        ('K', 'Capital', 'trillion $'),
        ('YGROSS', 'Gross Output', 'trillion $'),
        ('YNET', 'Net Output', 'trillion $'),
        ('ECO2', 'Total Emissions', 'GtCO2/yr'),
        ('TATM', 'Temperature', '°C'),
        ('DAMFRAC', 'Damage Fraction', '-'),
        ('CPC', 'Consumption per Capita', 'thousand $/yr'),
        ('L', 'Population', 'millions'),
        ('A', 'TFP', 'DICE units'),
        ('sigma', 'Carbon Intensity', 'kgCO2/$1000'),
        ('theta1', 'Backstop Price', '$/tCO2'),
    ]

    raw_rows = []
    for var, name, units in raw_vars:
        val = get_value_at_2020(bn_raw, var)
        raw_rows.append({
            'DICE Variable': var,
            'Description': name,
            'Value at 2020': val,
            'Units': units,
        })

    df_raw = pd.DataFrame(raw_rows)

    # Write to Excel
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df_comparison.to_excel(writer, sheet_name='Variables at 2020', index=False)
        df_params.to_excel(writer, sheet_name='Parameters', index=False)
        df_raw.to_excel(writer, sheet_name='DICE Raw Values', index=False)

    print(f"\nDone! Created {output_path}")
    print("\nSheets:")
    print("  - Variables at 2020: All model outputs compared at year 2020")
    print("  - Parameters: Model parameters comparison")
    print("  - DICE Raw Values: Original DICE values in their native units")

    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: Variables at 2020")
    print("="*80)
    print(df_comparison[['Variable', 'DICE (B&N 2023)', 'Our Forward Model', 'Difference (%)']].to_string(index=False))


if __name__ == '__main__':
    main()
