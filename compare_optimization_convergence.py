#!/usr/bin/env python3
"""
Compare optimization convergence across different max_evaluations settings.

Analyzes control variables (f and s) from config_011_*-f-*-f-f_* optimization results to compare:
- Statistics (mean, std, median) for different iteration counts
- RMS differences relative to best objective value baseline (greatest = least negative)
- Two time periods: [0, 400] and [5, 80]
- Groups results by flag patterns with independent baseline selection per group
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import re

def extract_config_info(directory_name):
    """
    Extract flag pattern, max_evaluations, and algorithm from directory name.

    Example: config_011_f-f-t-f-f_10_1_1000_el_20260106-070053
    Returns: ('f-f-t-f-f', 1000, 'SBPLX') or (None, None, None) if doesn't match filter criteria

    Example: config_011_f-f-t-f-f_10_1_1000_el_BOBYQA_20260106-070053
    Returns: ('f-f-t-f-f', 1000, 'BOBYQA')

    Only returns data for:
    - Pattern *-f-*-f-f (position 2 and 4 must be 'f', position 5 must be 'f')
    - fract_gdp = 1 (not 0.02)
    """
    # Pattern: config_011_{flags}_{n_points}_{fract_gdp}_{max_eval}_{opt}_[BOBYQA_]TIMESTAMP
    pattern = r'config_011_([tf]-[tf]-[tf]-[tf]-[tf])_\d+_([\d.]+)_(\d+k?)'
    match = re.search(pattern, directory_name)

    if match:
        flag_pattern = match.group(1)
        fract_gdp = float(match.group(2))
        max_eval_str = match.group(3)

        # Filter for _el_ directories only
        if '_el_' not in directory_name:
            return None, None, None

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

def parse_optimal_objective(output_dir):
    """
    Parse the optimal objective value from terminal_output.txt.

    Searches for the last occurrence of 'Objective value:' in the file
    to get the full precision value from iteration output. The iteration
    output shows unscaled values, so we multiply by OBJECTIVE_SCALE to
    get the properly scaled objective value.

    Returns the optimal objective value as a float, or None if not found.
    """
    from constants import OBJECTIVE_SCALE

    terminal_output_path = Path(output_dir) / 'terminal_output.txt'

    if not terminal_output_path.exists():
        return None

    last_objective = None
    with open(terminal_output_path, 'r') as f:
        for line in f:
            if 'Objective value:' in line:
                value_str = line.split(':')[1].strip()
                last_objective = float(value_str) * OBJECTIVE_SCALE

    return last_objective

def parse_optimization_summary(output_dir):
    """
    Parse optimization summary from terminal_output.txt.

    Returns a dict with:
        - iterations_performed: int
        - total_evaluations: int
        - final_control_points: int
        - total_runtime_seconds: float
    Returns None for any value not found.
    """
    terminal_output_path = Path(output_dir) / 'terminal_output.txt'

    if not terminal_output_path.exists():
        return {
            'iterations_performed': None,
            'total_evaluations': None,
            'final_control_points': None,
            'total_runtime_seconds': None
        }

    summary = {
        'iterations_performed': None,
        'total_evaluations': None,
        'final_control_points': None,
        'total_runtime_seconds': None
    }

    with open(terminal_output_path, 'r') as f:
        for line in f:
            if 'Iterations performed:' in line:
                value_str = line.split(':')[1].strip()
                summary['iterations_performed'] = int(value_str)
            elif 'Total evaluations:' in line:
                value_str = line.split(':')[1].strip()
                summary['total_evaluations'] = int(value_str)
            elif 'Final control points:' in line:
                value_str = line.split(':')[1].strip()
                summary['final_control_points'] = int(value_str)
            elif 'Total runtime:' in line:
                # Extract seconds from format: "Total runtime: 8814.53 seconds (146.91 minutes)"
                value_str = line.split(':')[1].strip()
                seconds_str = value_str.split()[0]  # Get "8814.53" before "seconds"
                summary['total_runtime_seconds'] = float(seconds_str)

    return summary

def load_optimization_results(output_dir='data/output'):
    """
    Load all optimization results from output directory.

    Returns
    -------
    dict
        Dictionary keyed by directory_name, each entry containing:
        - 'flag_pattern': str
        - 'algorithm': str
        - 'max_eval': int
        - 'df': DataFrame with optimization results
        - 'objective': float (optimal objective value)
        - 'summary': dict with iterations_performed, total_evaluations, etc.
    """
    all_runs = {}
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

        # Parse optimal objective value and optimization summary
        optimal_obj = parse_optimal_objective(csv_file.parent)
        summary = parse_optimization_summary(csv_file.parent)

        # Store with directory_name as unique key
        all_runs[directory_name] = {
            'flag_pattern': flag_pattern,
            'algorithm': algorithm,
            'max_eval': max_eval,
            'df': df,
            'objective': optimal_obj,
            'summary': summary,
        }
        print(f"  Loaded: {directory_name} - {flag_pattern}, {algorithm}, {max_eval} evals, obj={optimal_obj}")

    return all_runs

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

    # Load all results (keyed by directory name)
    all_runs = load_optimization_results()

    if not all_runs:
        print("ERROR: No results found!")
        sys.exit(1)

    # Time periods to analyze
    time_periods = {
        'Full [0-400]': (0, 400),
        'Early [5-80]': (5, 80),
    }

    # Get all unique flag patterns
    flag_patterns = sorted(set(run['flag_pattern'] for run in all_runs.values()))

    print(f"\nTotal runs loaded: {len(all_runs)}")
    print(f"Flag patterns: {flag_patterns}")

    # ========================================================================
    # Find best baseline for each flag pattern
    # Best = greatest (least negative) objective value
    # ========================================================================
    print("\n" + "="*80)
    print("SELECTING BASELINE FOR EACH FLAG PATTERN")
    print("="*80)

    baselines = {}
    for flag_pattern in flag_patterns:
        best_obj = -np.inf
        best_run_name = None

        for run_name, run_data in all_runs.items():
            if run_data['flag_pattern'] != flag_pattern:
                continue
            obj = run_data['objective']
            if obj is not None and obj > best_obj:
                best_obj = obj
                best_run_name = run_name

        if best_run_name is not None:
            baselines[flag_pattern] = {
                'run_name': best_run_name,
                'objective': best_obj,
                'df': all_runs[best_run_name]['df']
            }
            print(f"\n{flag_pattern}: {best_run_name} (obj = {best_obj:.6e})")
        else:
            print(f"\n{flag_pattern}: WARNING - No valid baseline found!")

    # ========================================================================
    # Summary table - process every run individually
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)

    # Create summary DataFrame
    summary_rows = []

    for run_name in sorted(all_runs.keys()):
        run_data = all_runs[run_name]
        flag_pattern = run_data['flag_pattern']
        algorithm = run_data['algorithm']
        max_eval = run_data['max_eval']
        df = run_data['df']
        obj_value = run_data['objective']
        summary = run_data['summary']

        for period_name, time_period in time_periods.items():
            stats = compute_statistics(df, time_period, variables=['f', 's'])

            # Get baseline info for this flag pattern
            baseline_info = baselines.get(flag_pattern)
            baseline_obj = baseline_info['objective'] if baseline_info else None
            baseline_df = baseline_info['df'] if baseline_info else None
            baseline_run_name = baseline_info['run_name'] if baseline_info else None

            # Compute objective departure (best - current, positive means worse than best)
            obj_departure = (baseline_obj - obj_value) if (baseline_obj is not None and obj_value is not None) else np.nan

            # Compute RMS vs baseline
            if baseline_df is not None:
                if run_name == baseline_run_name:
                    rms = {'f': 0.0, 's': 0.0}
                else:
                    rms = compute_rms_difference(df, baseline_df, time_period, variables=['f', 's'])
            else:
                rms = {'f': np.nan, 's': np.nan}

            # Add row for each variable
            for var in ['f', 's']:
                if var in stats:
                    s = stats[var]
                    summary_rows.append({
                        'period': period_name,
                        'flags': flag_pattern,
                        'algorithm': algorithm,
                        'max_eval': max_eval,
                        'run_name': run_name,
                        'variable': var,
                        'objective': obj_value,
                        'obj_departure': obj_departure,
                        'iterations_performed': summary['iterations_performed'],
                        'total_evaluations': summary['total_evaluations'],
                        'final_control_points': summary['final_control_points'],
                        'total_runtime_seconds': summary['total_runtime_seconds'],
                        'mean': s['mean'],
                        'std': s['std'],
                        'median': s['median'],
                        'rms_vs_best_baseline': rms.get(var, np.nan),
                    })

    summary_df = pd.DataFrame(summary_rows)

    if not summary_df.empty:
        # Sort by: period (desc for Full before Early), variable (asc), flags (asc), objective (desc)
        summary_df = summary_df.sort_values(
            by=['period', 'variable', 'flags', 'objective'],
            ascending=[False, True, True, False]
        ).reset_index(drop=True)

        # Print summary tables
        print("\nFull period [0-400]:")
        print("-" * 80)
        full_df = summary_df[summary_df['period'] == 'Full [0-400]']
        for var in ['f', 's']:
            print(f"\nVariable: {var}")
            var_df = full_df[full_df['variable'] == var][['flags', 'run_name', 'objective', 'obj_departure', 'mean', 'std', 'median', 'rms_vs_best_baseline']]
            print(var_df.to_string(index=False))

        print("\n\nEarly period [5-80]:")
        print("-" * 80)
        early_df = summary_df[summary_df['period'] == 'Early [5-80]']
        for var in ['f', 's']:
            print(f"\nVariable: {var}")
            var_df = early_df[early_df['variable'] == var][['flags', 'run_name', 'objective', 'obj_departure', 'mean', 'std', 'median', 'rms_vs_best_baseline']]
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
