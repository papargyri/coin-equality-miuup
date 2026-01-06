"""
Run forward integration using control trajectories from optimization results.

Loads configuration and optimized control variables from an output directory,
then runs a forward simulation with those exact controls. Useful for debugging
and verifying optimization results.

Usage:
    python run_integration.py <output_directory>

Arguments:
    output_directory: Path to optimization output directory containing JSON config
                     and results CSV file

Example:
    python run_integration.py data/output/config_010_t-t-t-f-f_10_1_1000_el_20260105-165554
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from parameters import load_configuration, ModelConfiguration
from economic_model import integrate_model, create_derived_variables
from output import save_results
from scipy.interpolate import PchipInterpolator


def load_control_trajectory_from_csv(csv_path):
    """
    Load control trajectories (f and s) from results CSV file.

    Parameters
    ----------
    csv_path : str
        Path to results CSV file

    Returns
    -------
    tuple
        (t_values, f_values, s_values) arrays
    """
    df = pd.read_csv(csv_path)

    # Extract column names (first part before comma)
    df.columns = [col.split(',')[0].strip() for col in df.columns]

    t_values = df['t'].values
    f_values = df['f'].values
    s_values = df['s'].values

    return t_values, f_values, s_values


def create_control_function_from_csv(t_values, f_values, s_values):
    """
    Create control function from CSV data using PCHIP interpolation.

    Parameters
    ----------
    t_values : array
        Time points
    f_values : array
        Abatement fraction values
    s_values : array
        Savings rate values

    Returns
    -------
    callable
        Control function f,s = control(t)
    """
    f_interpolator = PchipInterpolator(t_values, f_values, extrapolate=True)
    s_interpolator = PchipInterpolator(t_values, s_values, extrapolate=True)

    def control_function(t):
        f = f_interpolator(t)
        s = s_interpolator(t)
        return np.clip(f, 0.0, 1.0), np.clip(s, 0.0, 1.0)

    return control_function


def find_files_in_output_dir(output_dir):
    """
    Find JSON config and results CSV files in output directory.

    Parameters
    ----------
    output_dir : str
        Path to output directory

    Returns
    -------
    tuple
        (config_path, csv_path)
    """
    output_path = Path(output_dir)

    # Find JSON config file
    json_files = list(output_path.glob('*.json'))
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON config file found in {output_dir}")
    if len(json_files) > 1:
        print(f"Warning: Multiple JSON files found in {output_dir}:")
        for f in json_files:
            print(f"  {f.name}")
        print(f"Using: {json_files[0].name}")
    config_path = json_files[0]

    # Find results CSV file (want *_results.csv, not *_summary.csv)
    csv_files = [f for f in output_path.glob('*_results.csv')]
    if len(csv_files) == 0:
        # Try without _results suffix but exclude summary files
        csv_files = [f for f in output_path.glob('*.csv')
                     if 'summary' not in f.name]

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No results CSV file found in {output_dir}")
    if len(csv_files) > 1:
        print(f"Warning: Multiple CSV files found in {output_dir}:")
        for f in csv_files:
            print(f"  {f.name}")
        print(f"Using: {csv_files[0].name}")
    csv_path = csv_files[0]

    return str(config_path), str(csv_path)


def main():
    # Get output directory from command line (required)
    if len(sys.argv) != 2:
        print("Usage: python run_integration.py <output_directory>")
        print("Example: python run_integration.py data/output/config_010_t-t-t-f-f_10_1_1000_el_20260105-165554")
        sys.exit(1)

    output_dir = sys.argv[1]

    if not os.path.isdir(output_dir):
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)

    print(f'=' * 80)
    print(f'COIN_equality Forward Integration from Optimized Controls')
    print(f'=' * 80)
    print(f'Source directory: {output_dir}\n')

    # Find config and CSV files
    print('Searching for files...')
    config_path, csv_path = find_files_in_output_dir(output_dir)
    print(f'  Config file: {os.path.basename(config_path)}')
    print(f'  Results CSV: {os.path.basename(csv_path)}')

    # Load configuration
    print(f'\nLoading configuration...')
    config = load_configuration(config_path)
    print(f'  Run name: {config.run_name}')
    print(f'  Time span: {config.integration_params.t_start} to {config.integration_params.t_end} yr')
    print(f'  Time step: {config.integration_params.dt} yr')

    # Load control trajectories from CSV
    print(f'\nLoading control trajectories from CSV...')
    t_csv, f_csv, s_csv = load_control_trajectory_from_csv(csv_path)
    print(f'  Time points: {len(t_csv)}')
    print(f'  f range: [{f_csv.min():.6f}, {f_csv.max():.6f}]')
    print(f'  s range: [{s_csv.min():.6f}, {s_csv.max():.6f}]')

    # Show sample values
    print(f'\n  Sample control values:')
    sample_indices = [0, len(t_csv)//4, len(t_csv)//2, 3*len(t_csv)//4, -1]
    for idx in sample_indices:
        print(f'    t={t_csv[idx]:6.1f} yr: f={f_csv[idx]:.6f}, s={s_csv[idx]:.6f}')

    # Create control function
    print(f'\nCreating control function using PCHIP interpolation...')
    control_function = create_control_function_from_csv(t_csv, f_csv, s_csv)

    # Create new configuration with loaded controls
    forward_config = ModelConfiguration(
        run_name=f"{config.run_name}_forward",
        scalar_params=config.scalar_params,
        time_functions=config.time_functions,
        integration_params=config.integration_params,
        optimization_params=config.optimization_params,
        initial_state=config.initial_state,
        control_function=control_function
    )

    # Run integration
    print(f'\nRunning forward integration...')
    results = integrate_model(forward_config)
    results = create_derived_variables(results)

    # Display results
    print(f'\nIntegration complete!')
    print(f'Number of time points: {len(results["t"])}')

    print(f'\n' + '=' * 80)
    print(f'Results Summary')
    print(f'=' * 80)

    print(f'\nInitial State (t={results["t"][0]:.1f} yr):')
    print(f'  Capital stock (K):            {results["K"][0]:.3e} $')
    print(f'  Cumulative emissions (Ecum):  {results["Ecum"][0]:.3e} tCO2')
    print(f'  Temperature change (ΔT):      {results["delta_T"][0]:.3f} °C')
    print(f'  Gini index:                   {results["Gini"][0]:.4f}')
    print(f'  Gini consumption:             {results["gini_consumption"][0]:.4f}')
    print(f'  Mean utility (U):             {results["U"][0]:.6f}')

    print(f'\nFinal State (t={results["t"][-1]:.1f} yr):')
    print(f'  Capital stock (K):            {results["K"][-1]:.3e} $')
    print(f'  Cumulative emissions (Ecum):  {results["Ecum"][-1]:.3e} tCO2')
    print(f'  Temperature change (ΔT):      {results["delta_T"][-1]:.3f} °C')
    print(f'  Gini index:                   {results["Gini"][-1]:.4f}')
    print(f'  Gini consumption:             {results["gini_consumption"][-1]:.4f}')
    print(f'  Mean utility (U):             {results["U"][-1]:.6f}')

    # Check for unusual behavior
    print(f'\n' + '=' * 80)
    print(f'Diagnostics')
    print(f'=' * 80)

    gini_cons = results["gini_consumption"]
    gini_cons_range = gini_cons.max() - gini_cons.min()
    gini_cons_std = np.std(gini_cons)

    print(f'\nGini consumption statistics:')
    print(f'  Mean:  {gini_cons.mean():.6f}')
    print(f'  Std:   {gini_cons_std:.6f}')
    print(f'  Min:   {gini_cons.min():.6f} at t={results["t"][gini_cons.argmin()]:.1f} yr')
    print(f'  Max:   {gini_cons.max():.6f} at t={results["t"][gini_cons.argmax()]:.1f} yr')
    print(f'  Range: {gini_cons_range:.6f}')

    if gini_cons_std > 0.01:
        print(f'\n  WARNING: Large variation in gini_consumption (std={gini_cons_std:.6f})')
        print(f'  This may indicate numerical issues or policy switching.')

    # Check for discontinuities
    gini_cons_diff = np.diff(gini_cons)
    max_jump = np.abs(gini_cons_diff).max()
    max_jump_idx = np.abs(gini_cons_diff).argmax()

    print(f'\nGini consumption jumps:')
    print(f'  Max jump: {gini_cons_diff[max_jump_idx]:.6f} at t={results["t"][max_jump_idx]:.1f} yr')
    if max_jump > 0.001:
        print(f'\n  WARNING: Large jump in gini_consumption ({gini_cons_diff[max_jump_idx]:.6f})')
        print(f'  Time: t={results["t"][max_jump_idx]:.1f} → {results["t"][max_jump_idx+1]:.1f} yr')
        print(f'  Values: {gini_cons[max_jump_idx]:.6f} → {gini_cons[max_jump_idx+1]:.6f}')

    # Save results
    print(f'\n' + '=' * 80)
    print(f'Saving Results')
    print(f'=' * 80)
    plot_short_horizon = config.integration_params.plot_short_horizon
    config_filename_base = os.path.basename(config_path)
    output_paths = save_results(results, forward_config.run_name, plot_short_horizon,
                                config_filename=config_filename_base)
    print(f'Output directory: {output_paths["output_dir"]}')
    print(f'CSV file:         {output_paths["csv_file"]}')
    print(f'PDF file:         {output_paths["pdf_file"]}')
    if 'pdf_file_short' in output_paths:
        print(f'Short-term PDF:   {output_paths["pdf_file_short"]}')

    print(f'\n' + '=' * 80)
    print(f'Forward Integration Complete')
    print(f'=' * 80)


if __name__ == '__main__':
    main()
