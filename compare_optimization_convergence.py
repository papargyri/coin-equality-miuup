#!/usr/bin/env python3
"""
Compare optimization convergence across different max_evaluations settings.

Analyzes control variables (f and s) from optimization results to compare:
- Statistics (mean, std, median) for different iteration counts
- RMS differences relative to 50k iteration baseline
- Two time periods: [0, 400] and [5, 80]
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_config_info(directory_name):
    """
    Extract flag pattern, max_evaluations, and algorithm from directory name.

    Example: config_010_f-f-t-f-f_10_1_1000_el_20260106-070053
    Returns: ('f-f-t-f-f', 1000, 'SBPLX') or (None, None, None) if doesn't match filter criteria

    Example: config_010_f-f-t-f-f_10_1_1000_el_BOBYQA_20260106-070053
    Returns: ('f-f-t-f-f', 1000, 'BOBYQA')

    Only returns data for:
    - Pattern *-f-*-f-f (position 2 and 4 must be 'f', position 5 must be 'f')
    - fract_gdp = 1 (not 0.02)
    """
    # Pattern: config_XXX_{flags}_{n_points}_{fract_gdp}_{max_eval}_{opt}_[BOBYQA_]TIMESTAMP
    pattern = r'config_\d+_([tf]-[tf]-[tf]-[tf]-[tf])_\d+_([\d.]+)_(\d+k?)'
    match = re.search(pattern, directory_name)

    if match:
        flag_pattern = match.group(1)
        fract_gdp = float(match.group(2))
        max_eval_str = match.group(3)

        # Filter for *-f-*-f-f pattern (position 2, 4, 5 must be 'f')
        flags = flag_pattern.split('-')
        if flags[1] != 'f' or flags[3] != 'f' or flags[4] != 'f':
            return None, None, None

        # Filter for fract_gdp = 1 (not 0.02)
        if abs(fract_gdp - 1.0) > 0.01:  # Allow small floating point error
            return None, None, None

        # Convert k suffix to thousands
        if max_eval_str.endswith('k'):
            max_eval = int(max_eval_str[:-1]) * 1000
        else:
            max_eval = int(max_eval_str)

        # Detect algorithm
        algorithm = 'BOBYQA' if '_BOBYQA' in directory_name else 'SBPLX'

        return flag_pattern, max_eval, algorithm

    return None, None, None

def load_optimization_results(output_dir='data/output'):
    """
    Load all optimization results from output directory.

    Returns
    -------
    dict
        Nested dict: results[flag_pattern][algorithm][max_eval] = DataFrame
    """
    results = {}
    output_path = Path(output_dir)

    # Find all optimization_results.csv files
    csv_files = list(output_path.glob('*/*_optimization_results.csv'))

    print(f"Found {len(csv_files)} optimization results files")

    for csv_file in csv_files:
        directory_name = csv_file.parent.name
        flag_pattern, max_eval, algorithm = extract_config_info(directory_name)

        if flag_pattern is None:
            print(f"Warning: Could not parse {directory_name}")
            continue

        # Read CSV and clean column names
        df = pd.read_csv(csv_file)
        # Column names are in format "variable_name, Description, (units)"
        # Extract just the variable name
        df.columns = [col.split(',')[0].strip() for col in df.columns]

        # Store in nested dict
        if flag_pattern not in results:
            results[flag_pattern] = {}
        if algorithm not in results[flag_pattern]:
            results[flag_pattern][algorithm] = {}

        results[flag_pattern][algorithm][max_eval] = df
        print(f"  Loaded: {flag_pattern}, {algorithm}, {max_eval} evals - {len(df)} timesteps")

    return results

def compute_statistics(df, time_period, variables=['f', 's']):
    """
    Compute mean, std, median for variables in given time period.

    Parameters
    ----------
    df : DataFrame
        Results with 't', 'f', 's' columns
    time_period : tuple
        (t_min, t_max) for filtering
    variables : list
        Variables to analyze

    Returns
    -------
    dict
        {variable: {'mean': float, 'std': float, 'median': float}}
    """
    t_min, t_max = time_period
    mask = (df['t'] >= t_min) & (df['t'] <= t_max)
    df_filtered = df[mask]

    stats = {}
    for var in variables:
        if var in df_filtered.columns:
            values = df_filtered[var].values
            stats[var] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
            }
        else:
            stats[var] = {'mean': np.nan, 'std': np.nan, 'median': np.nan}

    return stats

def compute_rms_difference(df1, df2, time_period, variables=['f', 's']):
    """
    Compute RMS difference between two DataFrames for given time period.

    Parameters
    ----------
    df1, df2 : DataFrame
        Results to compare (must have same time points)
    time_period : tuple
        (t_min, t_max) for filtering
    variables : list
        Variables to compare

    Returns
    -------
    dict
        {variable: rms_difference}
    """
    t_min, t_max = time_period

    # Filter both DataFrames
    mask1 = (df1['t'] >= t_min) & (df1['t'] <= t_max)
    mask2 = (df2['t'] >= t_min) & (df2['t'] <= t_max)

    df1_filtered = df1[mask1].sort_values('t').reset_index(drop=True)
    df2_filtered = df2[mask2].sort_values('t').reset_index(drop=True)

    # Ensure same time points (interpolate if needed)
    if not np.allclose(df1_filtered['t'].values, df2_filtered['t'].values):
        print(f"Warning: Time points don't match, interpolating...")
        # Use df1 time points as reference
        t_ref = df1_filtered['t'].values
        rms = {}
        for var in variables:
            if var in df1_filtered.columns and var in df2_filtered.columns:
                val1 = df1_filtered[var].values
                val2 = np.interp(t_ref, df2_filtered['t'].values, df2_filtered[var].values)
                rms[var] = np.sqrt(np.mean((val1 - val2)**2))
            else:
                rms[var] = np.nan
        return rms

    rms = {}
    for var in variables:
        if var in df1_filtered.columns and var in df2_filtered.columns:
            diff = df1_filtered[var].values - df2_filtered[var].values
            rms[var] = np.sqrt(np.mean(diff**2))
        else:
            rms[var] = np.nan

    return rms

def main():
    """Main analysis function."""
    print("="*80)
    print("OPTIMIZATION CONVERGENCE ANALYSIS")
    print("="*80)

    # Load all results
    results = load_optimization_results()

    if not results:
        print("ERROR: No results found!")
        sys.exit(1)

    # Time periods to analyze
    time_periods = {
        'Full [0-400]': (0, 400),
        'Early [5-80]': (5, 80),
    }

    # Get all flag patterns, algorithms, and max_eval values
    flag_patterns = sorted(results.keys())
    all_algorithms = set()
    all_max_evals = set()
    for flag_data in results.values():
        all_algorithms.update(flag_data.keys())
        for algo_data in flag_data.values():
            all_max_evals.update(algo_data.keys())
    algorithms = sorted(all_algorithms)
    max_evals = sorted(all_max_evals)

    print(f"\nFlag patterns: {flag_patterns}")
    print(f"Algorithms: {algorithms}")
    print(f"Max evaluations: {max_evals}")

    # ========================================================================
    # Part 1: Statistics for each configuration
    # ========================================================================
    print("\n" + "="*80)
    print("STATISTICS FOR EACH CONFIGURATION")
    print("="*80)

    for period_name, time_period in time_periods.items():
        print(f"\n{period_name}:")
        print("-" * 80)

        for flag_pattern in flag_patterns:
            print(f"\nFlag pattern: {flag_pattern}")

            for algorithm in algorithms:
                if algorithm not in results[flag_pattern]:
                    continue

                print(f"  Algorithm: {algorithm}")

                for max_eval in max_evals:
                    if max_eval not in results[flag_pattern][algorithm]:
                        continue

                    df = results[flag_pattern][algorithm][max_eval]
                    stats = compute_statistics(df, time_period, variables=['f', 's'])

                    print(f"    {max_eval:6d} evals:")
                    for var in ['f', 's']:
                        if var in stats:
                            s = stats[var]
                            print(f"      {var}: mean={s['mean']:8.5f}, std={s['std']:8.5f}, median={s['median']:8.5f}")

    # ========================================================================
    # Part 2: RMS differences relative to 50k SBPLX baseline
    # ========================================================================
    print("\n" + "="*80)
    print("RMS DIFFERENCES RELATIVE TO 50k SBPLX BASELINE")
    print("="*80)

    baseline_eval = 50000  # Use 50k as baseline
    baseline_algo = 'SBPLX'  # Use SBPLX as baseline algorithm

    for period_name, time_period in time_periods.items():
        print(f"\n{period_name}:")
        print("-" * 80)

        for flag_pattern in flag_patterns:
            print(f"\nFlag pattern: {flag_pattern}")

            # Check if we have baseline data
            if baseline_algo not in results[flag_pattern] or baseline_eval not in results[flag_pattern][baseline_algo]:
                print(f"  WARNING: No {baseline_algo} {baseline_eval} eval baseline found!")
                continue

            baseline_df = results[flag_pattern][baseline_algo][baseline_eval]

            for algorithm in algorithms:
                if algorithm not in results[flag_pattern]:
                    continue

                print(f"  Algorithm: {algorithm}")

                for max_eval in max_evals:
                    # Skip comparison with baseline itself
                    if algorithm == baseline_algo and max_eval == baseline_eval:
                        continue

                    if max_eval not in results[flag_pattern][algorithm]:
                        continue

                    df = results[flag_pattern][algorithm][max_eval]
                    rms = compute_rms_difference(df, baseline_df, time_period, variables=['f', 's'])

                    print(f"    {max_eval:6d} evals vs {baseline_algo} {baseline_eval:6d}:")
                    for var in ['f', 's']:
                        if var in rms:
                            print(f"      {var}: RMS diff = {rms[var]:10.7f}")

    # ========================================================================
    # Part 3: Summary table
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    # Create summary DataFrame
    summary_rows = []

    for period_name, time_period in time_periods.items():
        for flag_pattern in flag_patterns:
            for algorithm in algorithms:
                if algorithm not in results[flag_pattern]:
                    continue

                for max_eval in max_evals:
                    if max_eval not in results[flag_pattern][algorithm]:
                        continue

                    df = results[flag_pattern][algorithm][max_eval]
                    stats = compute_statistics(df, time_period, variables=['f', 's'])

                    # Add statistics
                    for var in ['f', 's']:
                        if var in stats:
                            s = stats[var]
                            summary_rows.append({
                                'period': period_name,
                                'flags': flag_pattern,
                                'algorithm': algorithm,
                                'max_eval': max_eval,
                                'variable': var,
                                'mean': s['mean'],
                                'std': s['std'],
                                'median': s['median'],
                            })

                    # Add RMS differences (all algorithms compared against SBPLX 50k)
                    if baseline_algo in results[flag_pattern] and baseline_eval in results[flag_pattern][baseline_algo]:
                        if not (algorithm == baseline_algo and max_eval == baseline_eval):
                            baseline_df = results[flag_pattern][baseline_algo][baseline_eval]
                            rms = compute_rms_difference(df, baseline_df, time_period, variables=['f', 's'])

                            for var in ['f', 's']:
                                if var in rms:
                                    # Find the corresponding stats row and add RMS
                                    for row in summary_rows:
                                        if (row['period'] == period_name and
                                            row['flags'] == flag_pattern and
                                            row['algorithm'] == algorithm and
                                            row['max_eval'] == max_eval and
                                            row['variable'] == var):
                                            row['rms_vs_50k_SBPLX'] = rms[var]
                                            break

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        # Pivot for better viewing
        print("\nFull period [0-400]:")
        print("-" * 80)
        full_df = summary_df[summary_df['period'] == 'Full [0-400]']
        for var in ['f', 's']:
            print(f"\nVariable: {var}")
            var_df = full_df[full_df['variable'] == var][['flags', 'algorithm', 'max_eval', 'mean', 'std', 'median', 'rms_vs_50k_SBPLX']]
            print(var_df.to_string(index=False))

        print("\n\nEarly period [5-80]:")
        print("-" * 80)
        early_df = summary_df[summary_df['period'] == 'Early [5-80]']
        for var in ['f', 's']:
            print(f"\nVariable: {var}")
            var_df = early_df[early_df['variable'] == var][['flags', 'algorithm', 'max_eval', 'mean', 'std', 'median', 'rms_vs_50k_SBPLX']]
            print(var_df.to_string(index=False))

        # Save to CSV
        output_file = 'optimization_convergence_analysis.csv'
        summary_df.to_csv(output_file, index=False)
        print(f"\n\nSummary saved to: {output_file}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()
