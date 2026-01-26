#!/usr/bin/env python3
"""
Create Excel workbook comparing DICE model parameters and forward model results.

This generates an xlsx file with:
1. Parameters sheet - Key model parameters from both models
2. Time series sheets - Side-by-side comparison of all variables over time

Usage:
    python create_bn_comparison_xlsx.py [--output <path>]
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


def load_dice_parameters():
    """Load key parameters from DICE Excel file."""
    excel_path = Path('barrage_nordhaus_2023/DICE2023-Excel-b-4-3-10-v18.3.xlsx')
    df = pd.read_excel(excel_path, sheet_name='Opt', header=None)

    params = {}

    # Economic parameters
    params['Capital share (alpha)'] = float(df.iloc[4, 1])
    params['Depreciation rate (delta)'] = float(df.iloc[5, 1])
    params['Initial output (trillion $)'] = float(df.iloc[6, 1])
    params['Initial capital (trillion $)'] = float(df.iloc[7, 1])

    # Welfare parameters
    params['Time preference (rho)'] = float(df.iloc[9, 1])
    params['Elasticity of MU (eta)'] = float(df.iloc[10, 1])

    # Damage function
    params['Linear damage coeff (psi1)'] = float(df.iloc[18, 1])
    params['Quadratic damage coeff (psi2)'] = float(df.iloc[19, 1])
    params['Damage exponent'] = float(df.iloc[20, 1])

    # Abatement
    params['Abatement exponent (theta2)'] = float(df.iloc[26, 1])
    params['Backstop price 2050 ($/tCO2)'] = float(df.iloc[23, 1])

    # Population
    params['Initial population (millions)'] = float(df.iloc[37, 1])

    # Emissions
    params['Initial sigma (kgCO2/$1000)'] = float(df.iloc[28, 1])
    params['Initial land-use emissions (GtCO2/yr)'] = float(df.iloc[31, 1])

    # Initial TFP
    params['Initial TFP'] = float(df.iloc[14, 1])

    return params


def create_parameters_sheet(dice_params):
    """Create parameters comparison dataframe."""
    # Our model parameters (from the forward run config)
    our_params = {
        'Capital share (alpha)': 0.3,
        'Depreciation rate (delta)': 0.1,
        'Initial output (trillion $)': 134.89,  # From Y_gross at 2020
        'Initial capital (trillion $)': 295.0,
        'Time preference (rho)': 0.026341,  # Used in forward run
        'Elasticity of MU (eta)': 0.95,  # Used in forward run (DICE-like)
        'Linear damage coeff (psi1)': 0.0,
        'Quadratic damage coeff (psi2)': 0.003467,
        'Damage exponent': 2.0,
        'Abatement exponent (theta2)': 2.6,
        'Backstop price 2050 ($/tCO2)': 515.0,
        'Initial population (millions)': 7752.9,
        'Initial sigma (kgCO2/$1000)': 0.291355,
        'Initial land-use emissions (GtCO2/yr)': 0.0,  # Not included in our model
        'Initial TFP': 5.84,  # B&N units
    }

    data = []
    for param_name in dice_params.keys():
        dice_val = dice_params[param_name]
        our_val = our_params.get(param_name, 'N/A')
        if isinstance(our_val, (int, float)) and isinstance(dice_val, (int, float)):
            if dice_val != 0:
                diff_pct = 100 * (our_val - dice_val) / dice_val
            else:
                diff_pct = 0 if our_val == 0 else float('inf')
        else:
            diff_pct = 'N/A'

        data.append({
            'Parameter': param_name,
            'DICE (B&N 2023)': dice_val,
            'Our Forward Model': our_val,
            'Difference (%)': diff_pct if isinstance(diff_pct, str) else f'{diff_pct:.2f}%'
        })

    return pd.DataFrame(data)


def create_time_series_sheet(bn_data, forward_results, var_bn, var_ours, description):
    """Create time series comparison dataframe."""
    bn_years = np.array(bn_data['year'])
    our_years = np.array(forward_results['t'])

    # Get B&N values
    bn_values = np.array(bn_data[var_bn]) if var_bn in bn_data else [np.nan] * len(bn_years)

    # Get our values, interpolated to B&N time points
    if var_ours in forward_results:
        from scipy.interpolate import interp1d
        our_interp = interp1d(our_years, forward_results[var_ours],
                             kind='linear', fill_value='extrapolate')
        our_values = our_interp(bn_years)
    else:
        our_values = [np.nan] * len(bn_years)

    # Compute difference
    with np.errstate(divide='ignore', invalid='ignore'):
        diff = our_values - bn_values
        diff_pct = 100 * diff / np.abs(bn_values)
        diff_pct = np.where(np.isfinite(diff_pct), diff_pct, np.nan)

    df = pd.DataFrame({
        'Year': bn_years,
        f'DICE ({description})': bn_values,
        f'Ours ({description})': our_values,
        'Difference': diff,
        'Difference (%)': diff_pct
    })

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Create B&N comparison Excel workbook'
    )
    parser.add_argument('--output', type=str,
                       default='data/output/bn_comparison/dice_vs_forward_comparison.xlsx',
                       help='Output xlsx path')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading data...")

    # Load DICE parameters
    dice_params = load_dice_parameters()
    print(f"  Loaded {len(dice_params)} DICE parameters")

    # Load B&N results (converted units)
    bn_data = load_bn_results('optimal')
    print(f"  Loaded B&N results: {len(bn_data['year'])} time points")

    # Load B&N raw results (original units)
    bn_raw = load_bn_raw_results()

    # Load forward results
    forward_results = load_forward_results()
    print(f"  Loaded forward results: {len(forward_results['t'])} time points")

    print(f"\nCreating Excel workbook: {output_path}")

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Parameters sheet
        print("  Creating Parameters sheet...")
        params_df = create_parameters_sheet(dice_params)
        params_df.to_excel(writer, sheet_name='Parameters', index=False)

        # Time series sheets - converted units (our model units)
        time_series_vars = [
            ('MIU', 'mu', 'Emission Control Rate'),
            ('S', 's', 'Savings Rate'),
            ('K', 'K', 'Capital ($)'),
            ('Y_gross', 'Y_gross', 'Gross Output ($)'),
            ('Y_net', 'Y_net', 'Net Output ($)'),
            ('Ecum', 'Ecum', 'Cumulative Emissions (tCO2)'),
            ('E', 'E', 'Annual Emissions (tCO2/yr)'),
            ('delta_T', 'delta_T', 'Temperature (°C)'),
            ('Omega', 'Omega', 'Damage Fraction'),
            ('consumption', 'consumption', 'Consumption per capita ($)'),
        ]

        for bn_var, our_var, description in time_series_vars:
            print(f"  Creating {description} sheet...")
            # Clean sheet name: remove invalid chars and truncate to 31 chars (Excel limit)
            sheet_name = description.replace('/', '-').replace('$', '').replace('°', 'deg')[:31]
            df = create_time_series_sheet(bn_data, forward_results, bn_var, our_var, description)
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Raw DICE values sheet (original B&N units)
        print("  Creating DICE Raw Values sheet...")
        raw_vars = ['MIU', 'S', 'K', 'YGROSS', 'YNET', 'ECO2', 'TATM', 'DAMFRAC', 'CPC', 'L', 'A', 'sigma', 'theta1']
        raw_data = {'Year': bn_raw['year']}
        for var in raw_vars:
            if var in bn_raw:
                raw_data[var] = bn_raw[var]
        raw_df = pd.DataFrame(raw_data)
        raw_df.to_excel(writer, sheet_name='DICE Raw Values', index=False)

        # Summary statistics sheet
        print("  Creating Summary sheet...")
        summary_data = []
        for bn_var, our_var, description in time_series_vars:
            if bn_var not in bn_data or our_var not in forward_results:
                continue

            bn_vals = np.array(bn_data[bn_var])
            our_vals = np.array(forward_results[our_var])

            # Interpolate our values to B&N time points
            from scipy.interpolate import interp1d
            our_interp = interp1d(forward_results['t'], our_vals,
                                 kind='linear', fill_value='extrapolate')
            our_at_bn = our_interp(bn_data['year'])

            # Compute statistics at key years
            years = np.array(bn_data['year'])
            for year in [2020, 2050, 2100]:
                idx = np.argmin(np.abs(years - year))
                bn_val = bn_vals[idx]
                our_val = our_at_bn[idx]
                diff_pct = 100 * (our_val - bn_val) / bn_val if bn_val != 0 else 0

                summary_data.append({
                    'Variable': description,
                    'Year': int(year),
                    'DICE Value': bn_val,
                    'Our Value': our_val,
                    'Difference (%)': diff_pct
                })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\nDone! Created {output_path}")
    print(f"\nSheets in workbook:")
    print("  - Parameters: Model parameters comparison")
    print("  - Summary: Key years comparison (2020, 2050, 2100)")
    print("  - Time series sheets for each variable")
    print("  - DICE Raw Values: Original B&N values in their units")


if __name__ == '__main__':
    main()
