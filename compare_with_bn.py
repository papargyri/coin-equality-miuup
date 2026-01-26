#!/usr/bin/env python3
"""
Compare optimization results with Barrage & Nordhaus (2023) DICE results.

This script loads an existing comparison and adds the B&N optimal/baseline
results for side-by-side comparison.

Usage:
    python compare_with_bn.py [--comparison-dir <path>] [--output <path>]

Arguments:
    --comparison-dir: Path to existing comparison directory (default: most recent)
    --output: Output PDF path (default: creates in comparison dir)
    --bn-scenario: 'optimal' or 'baseline' (default: optimal)
    --t-min: Start year for plots (default: 2025)
    --t-max: End year for plots (default: 2100)
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_bn_results(scenario='optimal'):
    """Load B&N results from JSON file."""
    path = Path(f'barrage_nordhaus_2023/bn_results_{scenario}.json')
    if not path.exists():
        raise FileNotFoundError(
            f"B&N results not found at {path}. Run extract_bn_results.py first."
        )
    with open(path, 'r') as f:
        return json.load(f)


def load_forward_results():
    """Load our forward simulation results with B&N controls."""
    path = Path('data/output/bn_comparison/our_results_optimal.json')
    if not path.exists():
        raise FileNotFoundError(
            f"Forward results not found at {path}. Run run_forward_bn_comparison.py first."
        )
    with open(path, 'r') as f:
        return json.load(f)


def load_comparison_results(comparison_dir):
    """Load results from existing comparison xlsx."""
    xlsx_path = comparison_dir / 'results_comparison_summary.xlsx'
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Results file not found: {xlsx_path}")

    # Load all sheets
    xlsx = pd.ExcelFile(xlsx_path)
    results = {}

    # Get run names from Directories sheet
    dirs_df = pd.read_excel(xlsx, sheet_name='Directories')
    run_names = dirs_df['Case Name'].tolist()

    # Mapping from sheet names to variable names
    sheet_to_var = {
        'Temperature Change from Pre-Ind': 'delta_T',
        'CO2 Emissions Rate': 'E',
        'Cumulative CO2 Emissions': 'Ecum',
        'Capital Stock': 'K',
        'Gross Production': 'Y_gross',
        'Net Output After Damages & Abat': 'Y_net',
        'Total Consumption': 'Consumption',
        'Emissions Abatement Fraction': 'mu',
        'Savings Rate': 's',
        'Climate Damage (% of Output)': 'Omega',
        'Abatement Allocation Fraction': 'f',
    }

    for sheet_name, var_name in sheet_to_var.items():
        if sheet_name in xlsx.sheet_names:
            df = pd.read_excel(xlsx, sheet_name=sheet_name)
            results[var_name] = df

    return results, run_names


def create_combined_comparison_pdf(comparison_results, run_names, bn_data, forward_results,
                                    output_path, t_min=2025, t_max=2100):
    """
    Create comparison PDF with existing runs plus B&N data.

    Parameters
    ----------
    comparison_results : dict
        Dictionary of DataFrames from existing comparison
    run_names : list
        Names of existing runs
    bn_data : dict
        B&N results dictionary
    forward_results : dict
        Our forward simulation results with B&N controls
    output_path : str
        Path to save PDF
    t_min, t_max : float
        Time range for plots
    """
    # Variables to plot
    variables = [
        ('delta_T', 'Temperature Change', '°C above pre-industrial'),
        ('mu', 'Emission Control Rate (μ)', 'Fraction'),
        ('s', 'Savings Rate', 'Fraction'),
        ('Ecum', 'Cumulative CO2 Emissions', 'tCO2'),
        ('E', 'Annual CO2 Emissions', 'tCO2/year'),
        ('K', 'Capital Stock', '$'),
        ('Y_gross', 'Gross Output', '$'),
        ('Y_net', 'Net Output', '$'),
        ('Omega', 'Damage Fraction', 'Fraction'),
        ('Consumption', 'Total Consumption', '$'),
    ]

    # Simplify run names for legend
    def simplify_name(name):
        if 'mu_up_true' in name and '2.7' in name:
            return 'Ours: μ_up=True, ρ=2.7%'
        elif 'mu_up_true' in name and '_1_' in name:
            return 'Ours: μ_up=True, ρ=1%'
        elif 'mu_up_false' in name and '2.7' in name:
            return 'Ours: μ_up=False, ρ=2.7%'
        elif 'mu_up_false' in name and '_1_' in name:
            return 'Ours: μ_up=False, ρ=1%'
        else:
            # Truncate long names
            if len(name) > 30:
                return name[:27] + '...'
            return name

    # Colors for runs
    colors = plt.cm.tab10(np.linspace(0, 1, len(run_names) + 2))

    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.5, 0.6, 'Model Comparison', ha='center', va='center', fontsize=24, fontweight='bold')
        fig.text(0.5, 0.45, 'Our Model Runs vs Barrage & Nordhaus (2023) DICE', ha='center', va='center', fontsize=16)
        fig.text(0.5, 0.35, f'Time range: {t_min} - {t_max}', ha='center', va='center', fontsize=14)

        # Legend info
        legend_text = 'Runs compared:\n'
        for i, name in enumerate(run_names):
            legend_text += f'  • {simplify_name(name)}\n'
        legend_text += '  • B&N DICE Optimal (prescribed MIU & S)\n'
        legend_text += '  • Ours (Forward with B&N controls)'
        fig.text(0.5, 0.15, legend_text, ha='center', va='center', fontsize=10, family='monospace')

        pdf.savefig(fig)
        plt.close(fig)

        # B&N years for interpolation
        bn_years = np.array(bn_data['year'])

        # Plot each variable
        for var_name, title, ylabel in variables:
            if var_name not in comparison_results:
                continue

            fig, ax = plt.subplots(figsize=(12, 7))

            df = comparison_results[var_name]
            time_col = df.columns[0]  # First column is time
            times = df[time_col].values

            # Filter to time range
            mask = (times >= t_min) & (times <= t_max)
            times_filtered = times[mask]

            # Plot each existing run
            for i, run_name in enumerate(run_names):
                if run_name in df.columns:
                    values = df[run_name].values[mask]
                    label = simplify_name(run_name)
                    ax.plot(times_filtered, values, '-', color=colors[i],
                           linewidth=1.5, label=label, alpha=0.8)

            # Add B&N data
            bn_var = var_name
            if bn_var in bn_data:
                bn_values = np.array(bn_data[bn_var])
                # Interpolate to match time points
                bn_mask = (bn_years >= t_min) & (bn_years <= t_max)
                ax.plot(bn_years[bn_mask], bn_values[bn_mask], 'k--',
                       linewidth=2.5, label='B&N DICE Optimal', marker='o', markersize=4,
                       markevery=2)

            # Add our forward simulation results
            if var_name in forward_results:
                fwd_times = np.array(forward_results['t'])
                fwd_values = np.array(forward_results[var_name])
                fwd_mask = (fwd_times >= t_min) & (fwd_times <= t_max)
                ax.plot(fwd_times[fwd_mask], fwd_values[fwd_mask], 'r:',
                       linewidth=2, label='Ours (B&N controls)', alpha=0.9)

            ax.set_xlabel('Year', fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(t_min, t_max)

            # Use log scale for large quantities
            if var_name in ['K', 'Y_gross', 'Y_net', 'Consumption', 'Ecum']:
                if np.all(df[run_names[0]].values[mask] > 0):
                    ax.set_yscale('log')

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # Create difference plots for key variables
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Relative Difference: (Our Runs - B&N) / B&N × 100%', fontsize=14, fontweight='bold')

        key_vars = [('delta_T', 'Temperature'), ('Y_gross', 'Gross Output'),
                   ('Ecum', 'Cumulative Emissions'), ('mu', 'Emission Control Rate')]

        for idx, (var_name, title) in enumerate(key_vars):
            ax = axes[idx // 2, idx % 2]

            if var_name not in comparison_results or var_name not in bn_data:
                ax.text(0.5, 0.5, f'No data for {var_name}', ha='center', va='center')
                continue

            df = comparison_results[var_name]
            time_col = df.columns[0]
            times = df[time_col].values
            mask = (times >= t_min) & (times <= t_max)
            times_filtered = times[mask]

            bn_values = np.array(bn_data[var_name])
            # Interpolate B&N to our time points
            bn_interp = interp1d(bn_years, bn_values, kind='linear',
                                fill_value='extrapolate')(times_filtered)

            for i, run_name in enumerate(run_names):
                if run_name in df.columns:
                    our_values = df[run_name].values[mask]
                    with np.errstate(divide='ignore', invalid='ignore'):
                        rel_diff = 100 * (our_values - bn_interp) / np.abs(bn_interp)
                        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)
                    ax.plot(times_filtered, rel_diff, '-', color=colors[i],
                           linewidth=1.5, label=simplify_name(run_name), alpha=0.8)

            ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax.axhline(y=10, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axhline(y=-10, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Year')
            ax.set_ylabel('Difference (%)')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(t_min, t_max)
            ax.set_ylim(-50, 50)
            if idx == 0:
                ax.legend(loc='best', fontsize=8)

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"Comparison PDF saved to {output_path}")


def find_latest_comparison_dir():
    """Find the most recent comparison directory."""
    comparison_dirs = list(Path('data/output').glob('comparison_*'))
    if not comparison_dirs:
        raise FileNotFoundError("No comparison directories found in data/output/")

    # Sort by modification time, most recent first
    comparison_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return comparison_dirs[0]


def main():
    parser = argparse.ArgumentParser(
        description='Compare model results with B&N DICE'
    )
    parser.add_argument('--comparison-dir', type=str, default=None,
                       help='Path to existing comparison directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PDF path')
    parser.add_argument('--bn-scenario', choices=['optimal', 'baseline'],
                       default='optimal', help='B&N scenario to use')
    parser.add_argument('--t-min', type=float, default=2025,
                       help='Start year for plots')
    parser.add_argument('--t-max', type=float, default=2100,
                       help='End year for plots')
    args = parser.parse_args()

    # Find comparison directory
    if args.comparison_dir:
        comparison_dir = Path(args.comparison_dir)
    else:
        comparison_dir = find_latest_comparison_dir()

    print(f"Using comparison directory: {comparison_dir}")

    # Load existing comparison results
    print("Loading existing comparison results...")
    comparison_results, run_names = load_comparison_results(comparison_dir)
    print(f"  Found {len(run_names)} runs: {[n[:30] + '...' if len(n) > 30 else n for n in run_names]}")

    # Load B&N data
    print(f"Loading B&N {args.bn_scenario} results...")
    bn_data = load_bn_results(args.bn_scenario)

    # Load our forward simulation results
    print("Loading forward simulation results...")
    try:
        forward_results = load_forward_results()
    except FileNotFoundError:
        print("  Warning: Forward results not found, running without them")
        forward_results = {}

    # Create output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = comparison_dir / f'comparison_with_bn_{args.t_min}-{args.t_max}.pdf'

    # Create combined comparison
    print(f"Creating comparison PDF for {args.t_min}-{args.t_max}...")
    create_combined_comparison_pdf(
        comparison_results, run_names, bn_data, forward_results,
        output_path, args.t_min, args.t_max
    )

    print(f"\nDone! Output: {output_path}")


if __name__ == '__main__':
    main()
