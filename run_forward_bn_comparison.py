"""
Run forward simulation with Barrage & Nordhaus (2023) control values.

This script loads the optimal MIU and S trajectories from B&N's DICE model
and runs our model with those control values for comparison. This allows
us to diagnose discrepancies between the two models.

Two comparison modes are available:
1. "inverse_f": Compute f from B&N's MIU using the inverse abatement formula
2. "dice_like": Run in DICE-like mode (no redistribution, uniform damage, etc.)

Usage:
    python run_forward_bn_comparison.py [--mode inverse_f|dice_like] [--plot]
"""

import argparse
import json
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from parameters import (
    ModelConfiguration, ScalarParameters,
    IntegrationParameters, OptimizationParameters,
    create_f_and_s_control_from_single,
)
from economic_model import integrate_model, create_derived_variables


def load_bn_results(scenario='optimal'):
    """
    Load B&N results from JSON file.

    Parameters
    ----------
    scenario : str
        'optimal' or 'baseline'

    Returns
    -------
    dict
        Dictionary with time series arrays
    """
    path = Path(f'barrage_nordhaus_2023/bn_results_{scenario}.json')
    if not path.exists():
        raise FileNotFoundError(
            f"B&N results not found at {path}. "
            "Run extract_bn_results.py first."
        )

    with open(path, 'r') as f:
        return json.load(f)


def create_bn_time_functions(bn_data):
    """
    Create time functions from B&N data.

    Parameters
    ----------
    bn_data : dict
        B&N results dictionary

    Returns
    -------
    dict
        Dictionary of time functions suitable for our model
    """
    years = np.array(bn_data['year'])
    # Convert to relative time (t=0 at 2020)
    t_relative = years - 2020

    time_functions = {}

    # Create interpolation functions for key variables
    # Use piecewise linear interpolation

    # TFP (A)
    A = np.array(bn_data['A'])
    time_functions['A'] = interp1d(t_relative, A, kind='linear',
                                   fill_value=(A[0], A[-1]), bounds_error=False)

    # Population (L)
    L = np.array(bn_data['L'])
    time_functions['L'] = interp1d(t_relative, L, kind='linear',
                                   fill_value=(L[0], L[-1]), bounds_error=False)

    # Carbon intensity (sigma)
    sigma = np.array(bn_data['sigma'])
    time_functions['sigma'] = interp1d(t_relative, sigma, kind='linear',
                                       fill_value=(sigma[0], sigma[-1]), bounds_error=False)

    # Backstop price (theta1)
    theta1 = np.array(bn_data['theta1'])
    time_functions['theta1'] = interp1d(t_relative, theta1, kind='linear',
                                        fill_value=(theta1[0], theta1[-1]), bounds_error=False)

    # Savings rate (s) - from B&N's optimized S
    S = np.array(bn_data['S'])
    time_functions['s'] = interp1d(t_relative, S, kind='linear',
                                   fill_value=(S[0], S[-1]), bounds_error=False)

    # Emission control rate (MIU) - for computing f
    MIU = np.array(bn_data['MIU'])
    time_functions['MIU'] = interp1d(t_relative, MIU, kind='linear',
                                     fill_value=(MIU[0], MIU[-1]), bounds_error=False)

    # Damage fraction (Omega) - for computing f from MIU
    Omega = np.array(bn_data['Omega'])
    time_functions['Omega'] = interp1d(t_relative, Omega, kind='linear',
                                       fill_value=(Omega[0], Omega[-1]), bounds_error=False)

    # Gini - use very small constant (B&N doesn't have inequality)
    # Model requires Gini > 0 for Pareto-Lorenz, so use tiny value
    time_functions['gini'] = lambda _t: 1e-6  # Effectively no inequality

    return time_functions


def compute_f_from_mu(mu, sigma, theta1, theta2, fract_gdp, Omega):
    """
    Compute f from mu using the inverse abatement formula.

    The forward formula is:
        abateCost = f * fract_gdp * y_damaged
        mu = (abateCost * theta2 / (sigma * y_gross * theta1))^(1/theta2)

    Since y_damaged = y_gross * (1 - Omega), we get:
        mu^theta2 = f * fract_gdp * (1-Omega) * theta2 / (sigma * theta1)
        f = mu^theta2 * sigma * theta1 / (fract_gdp * (1-Omega) * theta2)

    Parameters
    ----------
    mu : float
        Emission control rate (0 to 1)
    sigma : float
        Carbon intensity (tCO2/$)
    theta1 : float
        Abatement cost coefficient ($/tCO2)
    theta2 : float
        Abatement cost exponent
    fract_gdp : float
        Fraction of GDP available for abatement
    Omega : float
        Damage fraction

    Returns
    -------
    float
        Abatement allocation fraction f
    """
    if mu <= 0:
        return 0.0
    if Omega >= 1:
        return 1.0  # Avoid division by zero

    f = (mu ** theta2) * sigma * theta1 / (fract_gdp * (1 - Omega) * theta2)
    return np.clip(f, 0.0, 1.0)


def create_f_control_from_bn(bn_time_functions, theta2, fract_gdp):
    """
    Create f control function by inverting B&N's MIU.

    Parameters
    ----------
    bn_time_functions : dict
        Time functions from B&N data
    theta2 : float
        Abatement cost exponent
    fract_gdp : float
        Fraction of GDP available

    Returns
    -------
    callable
        f(t) control function
    """
    def f_from_bn(t):
        mu = bn_time_functions['MIU'](t)
        sigma = bn_time_functions['sigma'](t)
        theta1 = bn_time_functions['theta1'](t)
        Omega = bn_time_functions['Omega'](t)

        f = compute_f_from_mu(mu, sigma, theta1, theta2, fract_gdp, Omega)
        return f

    return f_from_bn


def create_dice_like_config(bn_data):
    """
    Create a DICE-like configuration using B&N parameter values.

    This sets up our model to run as close to DICE as possible:
    - No redistribution
    - Uniform damage (no income-dependent damage)
    - Use B&N's time functions for A, L, sigma, theta1
    - Use B&N's K_initial and Ecum_initial
    - Use B&N's optimized MIU (via f inversion) and S

    Parameters
    ----------
    bn_data : dict
        B&N results dictionary

    Returns
    -------
    ModelConfiguration
        Configuration ready for integration
    """
    # Create time functions from B&N
    bn_time_functions = create_bn_time_functions(bn_data)

    # Scalar parameters - DICE-like settings
    # Use B&N's damage coefficients
    # B&N uses: DAMFRAC = a1*T + a2*T^a3 with a1=0, a2=0.003467, a3=2
    # But our model uses psi1, psi2 for: Omega = psi1*T + psi2*T^2
    # So psi1=0, psi2=0.003467 should match

    scalar_params = ScalarParameters(
        alpha=0.3,           # Capital share (same as B&N)
        delta=0.1,           # Depreciation rate (same as B&N)
        psi1=0.0,            # Linear damage coefficient (same as B&N a1)
        psi2=0.003467,       # Quadratic damage coefficient (same as B&N a2)
        y_damage_distribution_exponent=0.0,  # No income-dependent damage
        y_net_reference=1.0,  # Not used when exponent=0
        k_climate=5.6869e-13, # Temperature sensitivity

        eta=0.95,            # DICE uses 0.95 (we often use 1.45)
        rho=0.026341,        # B&N's risk-adjusted rate: exp(0.001 + 0.5*0.05) - 1 ≈ 0.0263

        fract_gdp=1.0,       # Full GDP available
        theta2=2.6,          # Same as B&N

        K_initial=bn_data['K'][0],        # B&N's initial capital
        Ecum_initial=2193031000000.0,     # Calibrated to match B&N's 2020 T=1.247°C (= 1.24715/k_climate)

        # Policy switches - DICE-like (no redistribution, no income-dependent)
        income_dependent_aggregate_damage=False,
        income_dependent_damage_distribution=False,
        income_dependent_tax_policy=False,
        income_redistribution=False,
        income_dependent_redistribution_policy=False,
        use_empirical_lorenz=False,  # Not used when Gini=0

        t_base=2020.0,
        use_mu_up=False,     # No mu_up cap (B&N enforces cap differently)
        mu_up_schedule=[[2020, 1.0]],
        cap_spending_mode="no_waste",
        use_emissions_additions=False,
        emissions_additions_schedule=[[2020, 0.0]],
    )

    # Integration parameters - match B&N's 5-year steps but use our 1-year dt
    integration_params = IntegrationParameters(
        t_start=2020.0,
        t_end=2420.0,
        dt=1.0,              # 1-year steps
        n_quad=16,           # Not really needed when Gini=0
        rtol=1e-6,
        atol=1e-9,
        plot_short_horizon=100.0,
    )

    # Optimization parameters (not used in forward mode)
    optimization_params = OptimizationParameters(
        max_evaluations_initial=100,
        max_evaluations_final=100,
        max_evaluations_time_points=100,
        optimization_iterations=1,
        initial_guess_f=0.1,
    )

    # Create control function from B&N's MIU and S
    f_control = create_f_control_from_bn(
        bn_time_functions, scalar_params.theta2, scalar_params.fract_gdp
    )

    # Combine f and s control functions
    control_function = create_f_and_s_control_from_single(
        f_control, bn_time_functions['s']
    )

    return ModelConfiguration(
        run_name='bn_comparison_dice_like',
        scalar_params=scalar_params,
        time_functions=bn_time_functions,
        integration_params=integration_params,
        optimization_params=optimization_params,
        initial_state=None,
        control_function=control_function,
    )


def compare_results(our_results, bn_data, output_path=None):
    """
    Compare our model results with B&N data.

    Parameters
    ----------
    our_results : dict
        Our model integration results
    bn_data : dict
        B&N results dictionary
    output_path : str, optional
        Path to save comparison PDF
    """
    # Get B&N time points and our time points
    bn_years = np.array(bn_data['year'])
    our_years = our_results['t']

    # Interpolate our results to B&N time points for comparison
    def interp_ours(var_name):
        if var_name in our_results:
            return interp1d(our_years, our_results[var_name],
                          kind='linear', fill_value='extrapolate')(bn_years)
        return None

    # Variables to compare
    comparisons = [
        ('K', 'K', 'Capital ($)', 1.0),
        ('Y_gross', 'Y_gross', 'Gross Output ($)', 1.0),
        ('Y_net', 'Y_net', 'Net Output ($)', 1.0),
        ('Ecum', 'Ecum', 'Cumulative Emissions (tCO2)', 1.0),
        ('delta_T', 'delta_T', 'Temperature Change (°C)', 1.0),
        ('Omega', 'Omega', 'Damage Fraction', 1.0),
        ('E', 'E', 'Annual Emissions (tCO2/yr)', 1.0),
        ('consumption', 'consumption', 'Consumption per capita ($)', 1.0),
        ('mu', 'MIU', 'Emission Control Rate (μ)', 1.0),
        ('s', 'S', 'Savings Rate', 1.0),
    ]

    # Print comparison table
    print("\n" + "="*100)
    print("COMPARISON: Our Model vs Barrage & Nordhaus (2023)")
    print("="*100)

    # Select key years for comparison
    key_years = [2020, 2030, 2050, 2070, 2100, 2150, 2200]
    key_indices = [np.argmin(np.abs(bn_years - y)) for y in key_years]

    for our_var, bn_var, label, scale in comparisons:
        our_vals = interp_ours(our_var)
        if our_vals is None:
            continue

        bn_vals = np.array(bn_data[bn_var]) if bn_var in bn_data else None
        if bn_vals is None:
            continue

        print(f"\n{label}:")
        print(f"  {'Year':>8} {'Ours':>15} {'B&N':>15} {'Diff':>12} {'Diff %':>10}")
        print("  " + "-"*65)

        for idx, year in zip(key_indices, key_years):
            if idx < len(bn_vals) and idx < len(our_vals):
                ours = our_vals[idx] * scale
                bn = bn_vals[idx] * scale
                diff = ours - bn
                pct = 100 * diff / bn if bn != 0 else 0
                print(f"  {year:>8} {ours:>15.4g} {bn:>15.4g} {diff:>12.4g} {pct:>9.2f}%")

    # Create comparison plots if output path provided
    if output_path:
        create_comparison_plots(our_results, bn_data, output_path)


def create_comparison_plots(our_results, bn_data, output_path):
    """
    Create comparison plots and save to PDF.

    Parameters
    ----------
    our_results : dict
        Our model integration results
    bn_data : dict
        B&N results dictionary
    output_path : str
        Path to save PDF
    """
    bn_years = np.array(bn_data['year'])
    our_years = our_results['t']

    # Plot configurations
    plot_configs = [
        # (our_var, bn_var, title, ylabel, log_scale)
        ('K', 'K', 'Capital Stock', 'Capital ($)', True),
        ('Y_gross', 'Y_gross', 'Gross Output', 'Output ($)', True),
        ('Y_net', 'Y_net', 'Net Output', 'Output ($)', True),
        ('Ecum', 'Ecum', 'Cumulative Emissions', 'Emissions (tCO2)', True),
        ('delta_T', 'delta_T', 'Temperature Change', 'Temperature (°C)', False),
        ('Omega', 'Omega', 'Damage Fraction', 'Fraction', False),
        ('E', 'E', 'Annual Emissions', 'Emissions (tCO2/yr)', True),
        ('mu', 'MIU', 'Emission Control Rate', 'μ', False),
        ('s', 'S', 'Savings Rate', 's', False),
        ('consumption', 'consumption', 'Consumption per Capita', '$/person', True),
    ]

    with PdfPages(output_path) as pdf:
        for our_var, bn_var, title, ylabel, log_scale in plot_configs:
            if our_var not in our_results or bn_var not in bn_data:
                continue

            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            # Top plot: Time series comparison
            ax1 = axes[0]
            ax1.plot(our_years, our_results[our_var], 'b-', label='Our Model', linewidth=1.5)
            ax1.plot(bn_years, bn_data[bn_var], 'r--', label='B&N DICE', linewidth=1.5)
            ax1.set_xlabel('Year')
            ax1.set_ylabel(ylabel)
            ax1.set_title(f'{title}: Comparison')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            if log_scale and np.all(np.array(our_results[our_var]) > 0):
                ax1.set_yscale('log')

            # Bottom plot: Relative difference
            ax2 = axes[1]
            # Interpolate to B&N time points
            our_interp = interp1d(our_years, our_results[our_var],
                                 kind='linear', fill_value='extrapolate')(bn_years)
            bn_vals = np.array(bn_data[bn_var])

            # Compute relative difference
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_diff = 100 * (our_interp - bn_vals) / np.abs(bn_vals)
                rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)

            ax2.plot(bn_years, rel_diff, 'g-', linewidth=1.5)
            ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            ax2.axhline(y=5, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
            ax2.axhline(y=-5, color='r', linestyle='--', linewidth=0.5, alpha=0.5)
            ax2.set_xlabel('Year')
            ax2.set_ylabel('Relative Difference (%)')
            ax2.set_title(f'{title}: (Ours - B&N) / B&N × 100%')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-50, 50)  # Limit y-axis to reasonable range

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nComparison plots saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run forward simulation with B&N control values'
    )
    parser.add_argument('--scenario', choices=['optimal', 'baseline'],
                       default='optimal', help='B&N scenario to use')
    parser.add_argument('--plot', action='store_true',
                       help='Generate comparison plots')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    args = parser.parse_args()

    print("="*70)
    print("FORWARD MODE: Running with Barrage & Nordhaus (2023) Control Values")
    print("="*70)

    # Load B&N results
    print(f"\nLoading B&N {args.scenario} scenario results...")
    bn_data = load_bn_results(args.scenario)
    print(f"  Time span: {bn_data['year'][0]} to {bn_data['year'][-1]}")
    print(f"  Number of periods: {len(bn_data['year'])}")

    # Create DICE-like configuration
    print("\nCreating DICE-like configuration...")
    config = create_dice_like_config(bn_data)

    print(f"  K_initial: ${config.scalar_params.K_initial:.2e}")
    print(f"  Ecum_initial: {config.scalar_params.Ecum_initial:.2e} tCO2")
    print(f"  eta: {config.scalar_params.eta}")
    print(f"  rho: {config.scalar_params.rho}")
    print(f"  psi2 (damage): {config.scalar_params.psi2}")

    # Run forward integration
    print("\nRunning forward integration...")
    results = integrate_model(config, store_detailed_output=True)
    results = create_derived_variables(results)

    # Compare with B&N
    output_pdf = None
    if args.plot:
        output_dir = Path(args.output) if args.output else Path('data/output')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_pdf = output_dir / f'bn_comparison_{args.scenario}.pdf'

    compare_results(results, bn_data, output_pdf)

    # Save our results for further analysis
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save key results to JSON
        results_to_save = {
            't': results['t'].tolist(),
            'K': results['K'].tolist(),
            'Ecum': results['Ecum'].tolist(),
            'Y_gross': results['Y_gross'].tolist(),
            'Y_net': results['Y_net'].tolist(),
            'delta_T': results['delta_T'].tolist(),
            'Omega': results['Omega'].tolist(),
            'E': results['E'].tolist(),
            'mu': results['mu'].tolist(),
            's': results['s'].tolist(),
            'consumption': results['consumption'].tolist(),
        }

        output_json = output_dir / f'our_results_{args.scenario}.json'
        with open(output_json, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nOur results saved to {output_json}")

    print("\nDone!")


if __name__ == '__main__':
    main()
