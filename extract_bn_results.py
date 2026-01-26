"""
Extract Barrage & Nordhaus (2023) DICE model results from Excel file.

This script reads the DICE2023 Excel workbook and extracts key time series
for comparison with our model. The results are saved to JSON for use by
run_forward_bn_comparison.py.

Output variables:
- year: Time points (2020, 2025, ..., 2420)
- MIU: Emission control rate (0 to 1+)
- S: Savings rate (fraction of gross output)
- K: Capital stock (trillion 2019$)
- YGROSS: Gross output before damages/abatement (trillion 2019$)
- YNET: Net output after damages and abatement (trillion 2019$)
- L: Population (millions)
- TATM: Atmospheric temperature (°C above preindustrial)
- ECO2: Total CO2 emissions (GtCO2/year)
- CCATOT: Cumulative CO2 emissions (GtC from 1765)
- CPC: Consumption per capita (thousands 2019$/year)
- DAMAGES: Climate damages (trillion 2019$)
- DAMFRAC: Damages as fraction of gross output
- A: Total factor productivity
- sigma: Carbon intensity (kgCO2 per $1000 output)
- theta1: Abatement cost coefficient (backstop price, $/tCO2)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def extract_row_data(df, row_idx, n_periods):
    """Extract numerical data from a row, handling mixed types."""
    row_data = df.iloc[row_idx, 1:n_periods+1].values
    return np.array([float(x) if pd.notna(x) else np.nan for x in row_data])


def extract_bn_results(excel_path, sheet_name='Opt'):
    """
    Extract key results from DICE2023 Excel file.

    Parameters
    ----------
    excel_path : str
        Path to DICE2023 Excel file
    sheet_name : str
        Sheet name to read ('Opt' for optimal, 'Base' for baseline)

    Returns
    -------
    dict
        Dictionary with time series arrays for key variables
    """
    print(f"Reading {excel_path}, sheet '{sheet_name}'...")
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)

    # Get years from row 1 (0-indexed)
    years_raw = df.iloc[1, 1:].values
    # Find where years become NaN or invalid
    n_periods = 0
    for val in years_raw:
        try:
            year = float(val)
            if np.isnan(year) or year < 2000 or year > 2500:
                break
            n_periods += 1
        except (ValueError, TypeError):
            break

    years = np.array([float(years_raw[i]) for i in range(n_periods)])
    print(f"Found {n_periods} time periods: {years[0]:.0f} to {years[-1]:.0f}")

    # Row indices for key variables (0-indexed)
    # These were identified from examining the Excel structure
    row_indices = {
        # Control variables
        'S': 68,           # Savings rate (optimized)
        'MIU': 71,         # Emissions Control Rate (GAMS from output)

        # State variables
        'K': 57,           # Capital ($trill, 2019$)

        # Output variables
        'YGROSS': 45,      # Output gross of abatement and damages
        'YNET': 51,        # Net output net of damages and abatement

        # Climate variables
        'TATM': 42,        # Atmospheric temperature (deg C above preind)
        'ECO2': 53,        # Total carbon emissions (GTCO2 per year)

        # Consumption
        'C': 59,           # Consumption ($trill per year)
        'CPC': 62,         # Consumption per capita ($thous per year)

        # Damages
        'DAMAGES': 47,     # Climate damages
        'DAMFRAC': 46,     # Total damage (% gross output)

        # Abatement
        'ABATECOST': 50,   # Abatement cost

        # Exogenous parameters
        'L': 37,           # Population (millions)
        'A': 14,           # Total factor productivity
        'sigma': 28,       # Sigma (industrial, MTCO2/$1000 2019 US$)
        'theta1': 24,      # Backstop price (2019$ per ton CO2)
    }

    results = {'year': years.tolist()}

    for var_name, row_idx in row_indices.items():
        try:
            data = extract_row_data(df, row_idx, n_periods)
            results[var_name] = data.tolist()

            # Verify by printing first/last values
            valid_data = data[~np.isnan(data)]
            if len(valid_data) > 0:
                print(f"  {var_name}: {valid_data[0]:.6g} -> {valid_data[-1]:.6g}")
            else:
                print(f"  {var_name}: all NaN")
        except Exception as e:
            print(f"  {var_name}: ERROR - {e}")
            results[var_name] = [np.nan] * n_periods

    # Also extract cumulative emissions from a different location
    # Row 98 has "Cum E, GtC (from 1765+base)"
    try:
        ccatot = extract_row_data(df, 98, n_periods)
        results['CCATOT_GtC'] = ccatot.tolist()
        print(f"  CCATOT_GtC: {ccatot[0]:.6g} -> {ccatot[-1]:.6g}")
    except Exception as e:
        print(f"  CCATOT_GtC: ERROR - {e}")

    return results


def convert_units(results):
    """
    Convert B&N units to our model's units.

    B&N uses:
    - Capital: trillion 2019$
    - Output: trillion 2019$
    - Population: millions
    - sigma: kgCO2 per $1000 = tCO2 per million $
    - Emissions: GtCO2/year
    - Cumulative emissions: GtC

    Our model uses:
    - Capital: $ (not trillion)
    - Output: $ per person (per capita)
    - Population: persons (not millions)
    - sigma: tCO2 per $
    - Emissions: tCO2/year
    - Cumulative emissions: tCO2
    """
    converted = {'year': results['year']}

    years = np.array(results['year'])
    n = len(years)

    # Population: millions -> persons
    L_millions = np.array(results['L'])
    L_persons = L_millions * 1e6
    converted['L'] = L_persons.tolist()

    # Capital: trillion$ -> $
    K_trill = np.array(results['K'])
    K_dollars = K_trill * 1e12
    converted['K'] = K_dollars.tolist()

    # Gross output: trillion$ -> $/person
    YGROSS_trill = np.array(results['YGROSS'])
    y_gross = (YGROSS_trill * 1e12) / L_persons
    converted['y_gross'] = y_gross.tolist()
    converted['Y_gross'] = (YGROSS_trill * 1e12).tolist()

    # Net output: trillion$ -> $/person
    YNET_trill = np.array(results['YNET'])
    y_net = (YNET_trill * 1e12) / L_persons
    converted['y_net'] = y_net.tolist()
    converted['Y_net'] = (YNET_trill * 1e12).tolist()

    # sigma: tCO2/$1000 -> tCO2/$
    # B&N uses "MTCO2/$1000" which is metric tonnes CO2 per $1000 (not megatonnes!)
    # To convert to tCO2/$: divide by 1000
    sigma_t_per_1000 = np.array(results['sigma'])
    sigma_t_per_dollar = sigma_t_per_1000 / 1000.0
    converted['sigma'] = sigma_t_per_dollar.tolist()

    # Emissions: GtCO2/year -> tCO2/year total, and per capita
    ECO2_Gt = np.array(results['ECO2'])
    E_total = ECO2_Gt * 1e9  # tCO2/year
    e_per_capita = E_total / L_persons
    converted['E'] = E_total.tolist()
    converted['e'] = e_per_capita.tolist()

    # Cumulative emissions: GtC -> tCO2
    # 1 GtC = 3.664 GtCO2 = 3.664e9 tCO2
    CCATOT_GtC = np.array(results['CCATOT_GtC'])
    Ecum_tCO2 = CCATOT_GtC * 3.664 * 1e9
    converted['Ecum'] = Ecum_tCO2.tolist()

    # Consumption: trillion$ -> $/person
    C_trill = np.array(results['C'])
    c_per_capita = (C_trill * 1e12) / L_persons
    converted['consumption'] = c_per_capita.tolist()
    converted['Consumption'] = (C_trill * 1e12).tolist()

    # CPC is already per capita in thousands, convert to $
    CPC_thous = np.array(results['CPC'])
    converted['consumption_check'] = (CPC_thous * 1000).tolist()

    # Damages: trillion$ -> $/person
    DAMAGES_trill = np.array(results['DAMAGES'])
    climate_damage = (DAMAGES_trill * 1e12) / L_persons
    converted['climate_damage'] = climate_damage.tolist()
    converted['Climate_damage'] = (DAMAGES_trill * 1e12).tolist()

    # Damage fraction (already dimensionless, but as percentage)
    DAMFRAC = np.array(results['DAMFRAC'])
    converted['Omega'] = DAMFRAC.tolist()  # Already fraction

    # Control variables (dimensionless)
    converted['MIU'] = results['MIU']
    converted['S'] = results['S']

    # Temperature (already in °C)
    converted['TATM'] = results['TATM']
    converted['delta_T'] = results['TATM']  # Alias

    # TFP and theta1 (need scaling for our units)
    # B&N TFP: Y = A * (L/1000)^(1-alpha) * K^alpha where K in trillion, L in millions
    # Our TFP: y = A * (K/L)^alpha where K in $, L in persons
    # So A_ours = A_BN * 1e6^(1-alpha) * 1e12^alpha / 1e6^alpha
    #           = A_BN * 1e6^(1-alpha) * 1e12^alpha / 1e6^alpha
    #           = A_BN * 1e6^(1-2*alpha) * 1e12^alpha
    # For alpha=0.3: 1e6^0.4 * 1e12^0.3 = 251.19 * 1995.26 = 501187
    # Actually let's compute directly
    A_BN = np.array(results['A'])
    alpha = 0.3
    # B&N: YGROSS = A * (L/1000)^(1-alpha) * K^alpha where K in trillion, L in million
    # Our: Y_gross = A * L^(1-alpha) * K^alpha where K in $, L in persons
    # y_gross = A * (K/L)^alpha
    # So let's derive A_ours from YGROSS, K, L:
    # Y_gross = A_ours * L^(1-alpha) * K^alpha
    # A_ours = Y_gross / (L^(1-alpha) * K^alpha)
    # Using B&N values (in our units):
    Y_gross_dollars = YGROSS_trill * 1e12
    A_ours = Y_gross_dollars / (L_persons**(1-alpha) * K_dollars**alpha)
    converted['A'] = A_ours.tolist()

    # theta1 (backstop price) - already in $/tCO2
    converted['theta1'] = results['theta1']

    # Abatement cost: trillion$ -> $/person
    if 'ABATECOST' in results:
        ABATECOST_trill = np.array(results['ABATECOST'])
        abateCost = (ABATECOST_trill * 1e12) / L_persons
        converted['abateCost_amount'] = abateCost.tolist()
        converted['AbateCost'] = (ABATECOST_trill * 1e12).tolist()

    return converted


def main():
    """Extract and save B&N results."""
    excel_path = Path('barrage_nordhaus_2023/DICE2023-Excel-b-4-3-10-v18.3.xlsx')

    if not excel_path.exists():
        print(f"Error: Excel file not found at {excel_path}")
        return

    # Extract from Opt sheet (optimal scenario)
    print("\n" + "="*60)
    print("Extracting OPTIMAL scenario results")
    print("="*60)
    results_opt = extract_bn_results(excel_path, 'Opt')

    # Convert units
    print("\nConverting units to our model's conventions...")
    converted_opt = convert_units(results_opt)

    # Save raw results
    output_path_raw = Path('barrage_nordhaus_2023/bn_results_raw.json')
    with open(output_path_raw, 'w') as f:
        json.dump(results_opt, f, indent=2)
    print(f"\nRaw results saved to {output_path_raw}")

    # Save converted results
    output_path = Path('barrage_nordhaus_2023/bn_results_optimal.json')
    with open(output_path, 'w') as f:
        json.dump(converted_opt, f, indent=2)
    print(f"Converted results saved to {output_path}")

    # Also extract from Base sheet for comparison
    print("\n" + "="*60)
    print("Extracting BASELINE scenario results")
    print("="*60)
    results_base = extract_bn_results(excel_path, 'Base')
    converted_base = convert_units(results_base)

    output_path_base = Path('barrage_nordhaus_2023/bn_results_baseline.json')
    with open(output_path_base, 'w') as f:
        json.dump(converted_base, f, indent=2)
    print(f"Baseline results saved to {output_path_base}")

    # Print summary comparison
    print("\n" + "="*60)
    print("Summary: Optimal vs Baseline at 2050 and 2100")
    print("="*60)

    years = np.array(converted_opt['year'])
    idx_2050 = np.argmin(np.abs(years - 2050))
    idx_2100 = np.argmin(np.abs(years - 2100))

    print(f"\n{'Variable':<25} {'2050 Opt':>12} {'2050 Base':>12} {'2100 Opt':>12} {'2100 Base':>12}")
    print("-" * 75)

    for var in ['MIU', 'S', 'delta_T', 'Omega', 'y_gross', 'y_net']:
        opt_2050 = converted_opt[var][idx_2050]
        base_2050 = converted_base[var][idx_2050]
        opt_2100 = converted_opt[var][idx_2100]
        base_2100 = converted_base[var][idx_2100]
        print(f"{var:<25} {opt_2050:>12.4g} {base_2050:>12.4g} {opt_2100:>12.4g} {base_2100:>12.4g}")


if __name__ == '__main__':
    main()
