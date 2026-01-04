"""
Generate 2-panel PDF figure showing utility ratios under different climate damage and taxation scenarios.

Panel A: Climate damage impact as function of Gini index and damage distribution exponent
Panel B: Progressive taxation impact as function of Gini index and mean tax rate

Computes utility ratios analytically using CRRA utility, Lorenz curves, and quadrature integration.

Usage (examples):
- PDF only:
    python plot_utility_ratios.py json/config_008_f-t-f-f-f_10_1_1000.json
- PDF + CSV export for both panels:
    python plot_utility_ratios.py json/config_008_f-t-f-f-f_10_1_1000.json --csv
- PDF + CSV + XLSX export for both panels (requires pandas):
    python plot_utility_ratios.py json/config_008_f-t-f-f-f_10_1_1000.json --csv --xlsx

Notes:
- The required positional argument is the JSON configuration file.
- All output files are written to a timestamped directory under data/output/.
- CSV and XLSX files are named {run_name}_panel_a.{csv,xlsx} and {run_name}_panel_b.{csv,xlsx}.
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.special import roots_legendre
from scipy.optimize import root_scalar
from pathlib import Path
from datetime import datetime

from parameters import load_configuration, evaluate_params_at_time
from distribution_utilities import (
    L_empirical_lorenz_derivative,
    L_pareto_derivative,
    L_empirical_lorenz,
    L_pareto,
)
from constants import EPSILON, LOOSE_EPSILON, EMPIRICAL_LORENZ_BASE_GINI

DELTA_T_REPRESENTATIVE = 2.0

GINI_RANGE_DAMAGE = (0.0, EMPIRICAL_LORENZ_BASE_GINI)
N_GINI_DAMAGE = 50
EXPONENT_RANGE = (0.0, 1.0)
N_EXPONENT = 50

GINI_RANGE_TAX = (0.0, EMPIRICAL_LORENZ_BASE_GINI)
N_GINI_TAX = 50
TAX_RATE_RANGE = (0.0, 0.2)
N_TAX_RATE = 50


def calculate_crra_utility(consumption, eta):
    """
    Calculate CRRA utility for given consumption and risk aversion.

    Parameters
    ----------
    consumption : float or ndarray
        Consumption level(s)
    eta : float
        Coefficient of relative risk aversion

    Returns
    -------
    float or ndarray
        Utility value(s)

    Notes
    -----
    For η = 1: U(c) = ln(max(c, EPSILON))
    For η ≠ 1: U(c) = max(c, EPSILON)^(1-η) / (1-η)
    """
    consumption_safe = np.maximum(consumption, EPSILON)

    if np.abs(eta - 1.0) < EPSILON:
        return np.log(consumption_safe)
    else:
        return (consumption_safe ** (1.0 - eta)) / (1.0 - eta)


def compute_aggregate_utility(gini, y_mean, eta, s, n_quad, use_empirical_lorenz):
    """
    Compute aggregate utility over income distribution.

    Parameters
    ----------
    gini : float
        Gini coefficient
    y_mean : float
        Mean income
    eta : float
        CRRA parameter
    s : float
        Savings rate
    n_quad : int
        Number of quadrature points
    use_empirical_lorenz : bool
        Whether to use empirical Lorenz curve

    Returns
    -------
    float
        Aggregate utility (population-weighted average)
    """
    xi, wi = roots_legendre(n_quad)
    Fi = (xi + 1.0) / 2.0
    Fwi = wi / 2.0

    if use_empirical_lorenz:
        dLdF = L_empirical_lorenz_derivative(Fi, gini)
    else:
        dLdF = L_pareto_derivative(Fi, gini)

    yi = y_mean * dLdF
    ci = (1.0 - s) * yi
    utility_yi = calculate_crra_utility(ci, eta)

    U_aggregate = np.sum(Fwi * utility_yi)

    return U_aggregate


def compute_aggregate_utility_from_income_array(yi, eta, s, Fwi):
    """
    Compute aggregate utility directly from income array.

    Parameters
    ----------
    yi : ndarray
        Income at each quadrature point
    eta : float
        CRRA parameter
    s : float
        Savings rate
    Fwi : ndarray
        Quadrature weights

    Returns
    -------
    float
        Aggregate utility (population-weighted average)
    """
    ci = (1.0 - s) * yi
    utility_yi = calculate_crra_utility(ci, eta)
    U_aggregate = np.sum(Fwi * utility_yi)
    return U_aggregate


def apply_climate_damage_to_income(y_base, Omega_base, damage_exponent, y_net_reference, Fi, Fwi, gini, use_empirical_lorenz, normalize_aggregate_damage=True):
    """
    Apply income-dependent climate damage to income distribution.

    Parameters
    ----------
    y_base : float
        Base mean income (before damage)
    Omega_base : float
        Base damage fraction (from temperature)
    damage_exponent : float
        Income-dependent damage exponent
    y_net_reference : float
        Reference income for damage scaling
    Fi : ndarray
        Quadrature points in F space
    gini : float
        Gini coefficient
    Fwi : ndarray
        Quadrature weights (same order as Fi)
    gini : float
        Gini coefficient
    use_empirical_lorenz : bool
        Lorenz curve formulation
    normalize_aggregate_damage : bool, optional
        If True, rescales income-dependent damages so aggregate damage equals Omega_base · y_base

    Returns
    -------
    ndarray
        Income at each quadrature point after climate damage
    """
    if use_empirical_lorenz:
        dLdF_yi = L_empirical_lorenz_derivative(Fi, gini)
    else:
        dLdF_yi = L_pareto_derivative(Fi, gini)

    y_lorenz_yi = y_base * dLdF_yi

    if damage_exponent < EPSILON:
        omega_yi = Omega_base * np.ones_like(Fi)
    else:
        omega_yi = Omega_base * (y_lorenz_yi / y_net_reference) ** (-damage_exponent)

    if normalize_aggregate_damage:
        damage_unscaled = np.sum(Fwi * y_lorenz_yi * omega_yi)
        target_damage = Omega_base * y_base
        scale = target_damage / max(damage_unscaled, EPSILON)
        omega_yi = omega_yi * scale

    omega_yi = np.clip(omega_yi, 0.0, 1.0 - LOOSE_EPSILON)
    y_damaged_yi = y_lorenz_yi * (1.0 - omega_yi)

    return y_damaged_yi


def apply_progressive_taxation(y_base, tax_rate_mean, gini, Fi, Fwi, use_empirical_lorenz):
    """
    Apply progressive taxation (tax top earners above Fmax).

    Parameters
    ----------
    y_base : float
        Mean income before tax
    tax_rate_mean : float
        Mean tax rate (fraction of GDP)
    gini : float
        Gini coefficient
    Fi : ndarray
        Quadrature points
    Fwi : ndarray
        Quadrature weights
    use_empirical_lorenz : bool
        Lorenz formulation

    Returns
    -------
    ndarray
        Post-tax income at each quadrature point
    """
    if use_empirical_lorenz:
        dLdF_yi = L_empirical_lorenz_derivative(Fi, gini)
    else:
        dLdF_yi = L_pareto_derivative(Fi, gini)

    y_lorenz_yi = y_base * dLdF_yi

    tax_target = tax_rate_mean * y_base

    def tax_revenue_residual(Fmax):
        if use_empirical_lorenz:
            L_Fmax = L_empirical_lorenz(Fmax, gini)
            dLdF_Fmax = L_empirical_lorenz_derivative(Fmax, gini)
        else:
            L_Fmax = L_pareto(Fmax, gini)
            dLdF_Fmax = L_pareto_derivative(Fmax, gini)

        revenue = y_base * ((1.0 - L_Fmax) - (1.0 - Fmax) * dLdF_Fmax)

        return revenue - tax_target

    bracket_low = EPSILON
    bracket_high = 1.0 - EPSILON

    residual_low = tax_revenue_residual(bracket_low)
    residual_high = tax_revenue_residual(bracket_high)

    if residual_low * residual_high > 0:
        if residual_low < 0:
            Fmax = bracket_high
        else:
            Fmax = bracket_low
    else:
        result = root_scalar(tax_revenue_residual, bracket=[bracket_low, bracket_high], method='brentq')
        Fmax = result.root

    if use_empirical_lorenz:
        dLdF_Fmax = L_empirical_lorenz_derivative(Fmax, gini)
    else:
        dLdF_Fmax = L_pareto_derivative(Fmax, gini)

    y_Fmax = y_base * dLdF_Fmax

    y_taxed_yi = np.where(Fi < Fmax, y_lorenz_yi, y_Fmax)

    return y_taxed_yi


def apply_uniform_taxation(y_base, tax_rate, gini, Fi, use_empirical_lorenz):
    """
    Apply uniform taxation (same rate for all).

    Parameters
    ----------
    y_base : float
        Mean income before tax
    tax_rate : float
        Uniform tax rate
    gini : float
        Gini coefficient
    Fi : ndarray
        Quadrature points
    use_empirical_lorenz : bool
        Lorenz formulation

    Returns
    -------
    ndarray
        Post-tax income at each quadrature point
    """
    if use_empirical_lorenz:
        dLdF_yi = L_empirical_lorenz_derivative(Fi, gini)
    else:
        dLdF_yi = L_pareto_derivative(Fi, gini)

    y_taxed_yi = y_base * dLdF_yi * (1.0 - tax_rate)

    return y_taxed_yi


def compute_damage_ratio_grid(gini_mesh, exponent_mesh, base_params, n_quad):
    """
    Compute utility ratios for climate damage sweep.

    For each (Gini, exponent) point:
    - Baseline: no damage
    - Reference: uniform damage (exponent=0) at current Gini
    - Test: income-dependent damage (current exponent) at current Gini
    - Ratio = ΔU(Gini, exp) / ΔU(Gini, exp=0)
            = (U_test - U_baseline) / (U_reference - U_baseline)

    This measures the additional harm from income-dependent damage
    (beyond what uniform damage would cause) at a given inequality level.

    Parameters
    ----------
    gini_mesh : ndarray
        2D array of Gini values
    exponent_mesh : ndarray
        2D array of damage exponent values
    base_params : dict
        Base economic parameters
    n_quad : int
        Number of quadrature points

    Returns
    -------
    ndarray
        2D array of utility ratios
    """
    y_mean = base_params['y_gross']
    y_net_reference = base_params['y_net_reference']
    eta = base_params['eta']
    s = base_params['s']
    Omega_base = base_params['Omega_base']
    use_empirical_lorenz = base_params['use_empirical_lorenz']

    xi, wi = roots_legendre(n_quad)
    Fi = (xi + 1.0) / 2.0
    Fwi = wi / 2.0

    ratio_grid = np.zeros_like(gini_mesh)

    n_rows, n_cols = gini_mesh.shape

    for i in range(n_rows):
        for j in range(n_cols):
            gini = max(gini_mesh[i, j], EPSILON)
            exponent = exponent_mesh[i, j]

            U_no_damage = compute_aggregate_utility(gini, y_mean, eta, s, n_quad, use_empirical_lorenz)

            y_uniform_damage_yi = apply_climate_damage_to_income(
                y_mean, Omega_base, 0.0, y_net_reference, Fi, Fwi, gini, use_empirical_lorenz
            )
            U_uniform_damage = compute_aggregate_utility_from_income_array(y_uniform_damage_yi, eta, s, Fwi)

            y_income_dependent_damage_yi = apply_climate_damage_to_income(
                y_mean, Omega_base, exponent, y_net_reference, Fi, Fwi, gini, use_empirical_lorenz
            )
            U_income_dependent_damage = compute_aggregate_utility_from_income_array(y_income_dependent_damage_yi, eta, s, Fwi)

            ratio_grid[i, j] = (U_income_dependent_damage - U_no_damage) / (U_uniform_damage - U_no_damage)

    return ratio_grid


def compute_tax_ratio_grid(gini_mesh, tax_rate_mesh, base_params, n_quad):
    """
    Compute utility ratios for progressive taxation sweep.

    Parameters
    ----------
    gini_mesh : ndarray
        2D array of Gini values
    tax_rate_mesh : ndarray
        2D array of mean tax rate values
    base_params : dict
        Base economic parameters
    n_quad : int
        Number of quadrature points

    Returns
    -------
    ndarray
        2D array of utility ratios
    """
    y_gross = base_params['y_gross']
    eta = base_params['eta']
    s = base_params['s']
    use_empirical_lorenz = base_params['use_empirical_lorenz']

    xi, wi = roots_legendre(n_quad)
    Fi = (xi + 1.0) / 2.0
    Fwi = wi / 2.0

    gini_baseline = EPSILON

    U_no_tax = compute_aggregate_utility(gini_baseline, y_gross, eta, s, n_quad, use_empirical_lorenz)

    ratio_grid = np.zeros_like(gini_mesh)

    n_rows, n_cols = gini_mesh.shape

    for i in range(n_rows):
        for j in range(n_cols):
            gini = max(gini_mesh[i, j], EPSILON)
            tax_rate = tax_rate_mesh[i, j]

            if tax_rate < EPSILON:
                ratio_grid[i, j] = 1.0
                continue

            y_uniform_tax_yi = apply_uniform_taxation(y_gross, tax_rate, gini, Fi, use_empirical_lorenz)
            U_uniform_tax = compute_aggregate_utility_from_income_array(y_uniform_tax_yi, eta, s, Fwi)

            y_progressive_tax_yi = apply_progressive_taxation(y_gross, tax_rate, gini, Fi, Fwi, use_empirical_lorenz)
            U_progressive_tax = compute_aggregate_utility_from_income_array(y_progressive_tax_yi, eta, s, Fwi)

            ratio_grid[i, j] = (U_progressive_tax - U_no_tax) / (U_uniform_tax - U_no_tax)

    return ratio_grid


def create_contour_plot(ax, X, Y, Z, xlabel, ylabel, title, colorbar_label, vmin=None, vmax=None):
    """
    Create a single contour plot on given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on
    X : ndarray
        2D meshgrid for x-axis
    Y : ndarray
        2D meshgrid for y-axis
    Z : ndarray
        2D array of values to contour
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str
        Plot title
    colorbar_label : str
        Label for colorbar
    vmin : float, optional
        Minimum value for color scale
    vmax : float, optional
        Maximum value for color scale
    """
    # Generate explicit levels if vmin/vmax specified to ensure colorbar respects bounds
    if vmin is not None or vmax is not None:
        z_min = vmin if vmin is not None else np.min(Z)
        z_max_data = vmax if vmax is not None else np.max(Z)
        # Round z_max up to nearest 0.1 (so 10 * z_max is an integer)
        z_max = np.ceil(z_max_data * 10) / 10
        # Create levels with 0.1 spacing
        levels = np.arange(z_min, z_max + 0.05, 0.1)
        contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis_r', extend='neither')
        # Use same levels for black contour lines to align with color boundaries
        ax.contour(X, Y, Z, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    else:
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis_r')
        ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label(colorbar_label, fontsize=10)

    ax.grid(True, alpha=0.2, linestyle='--')


def create_two_panel_figure(damage_data, tax_data, output_pdf):
    """
    Create 2-panel PDF figure with both ratio plots.

    Parameters
    ----------
    damage_data : dict
        Contains 'gini_mesh', 'exponent_mesh', 'ratio_grid' for Panel A
    tax_data : dict
        Contains 'gini_mesh', 'tax_rate_mesh', 'ratio_grid' for Panel B
    output_pdf : Path
        Output PDF file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    create_contour_plot(
        axes[0],
        damage_data['gini_mesh'],
        damage_data['exponent_mesh'],
        damage_data['ratio_grid'],
        'Gini Index',
        'Climate Damage Distribution Exponent',
        'Panel A: Climate Damage Impact on Utility',
        'Utility Ratio',
        vmin=1.0
    )

    create_contour_plot(
        axes[1],
        tax_data['gini_mesh'],
        tax_data['tax_rate_mesh'],
        tax_data['ratio_grid'],
        'Gini Index',
        'Mean Tax Rate',
        'Panel B: Progressive Taxation Impact on Utility',
        'Utility Ratio'
    )

    plt.tight_layout()

    with PdfPages(output_pdf) as pdf:
        pdf.savefig(fig)

    plt.close(fig)


def write_panel_a_csv(path, gini_mesh, exponent_mesh, ratio_grid):
    """Write Panel A grid with Gini as columns and damage exponents as rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Assume meshgrid produced with exponent varying along rows, gini along columns
    gini_vals = gini_mesh[0, :]
    exponent_vals = exponent_mesh[:, 0]

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['damage_exponent'] + list(gini_vals)
        writer.writerow(header)
        for exp_idx, exp_val in enumerate(exponent_vals):
            row = [exp_val] + list(ratio_grid[exp_idx, :])
            writer.writerow(row)


def write_panel_a_xlsx(path, gini_mesh, exponent_mesh, ratio_grid):
    try:
        import pandas as pd
    except ImportError:
        print('pandas not installed; skipping Panel A XLSX export')
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gini_vals = gini_mesh[0, :]
    exponent_vals = exponent_mesh[:, 0]

    data = {
        'damage_exponent': exponent_vals,
    }
    for col_idx, g_val in enumerate(gini_vals):
        data[g_val] = ratio_grid[:, col_idx]

    df = pd.DataFrame(data)
    df.to_excel(path, index=False)


def write_panel_b_csv(path, gini_mesh, tax_rate_mesh, ratio_grid):
    """Write Panel B grid with Gini as columns and tax rates as rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gini_vals = gini_mesh[0, :]
    tax_rate_vals = tax_rate_mesh[:, 0]

    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['tax_rate'] + list(gini_vals)
        writer.writerow(header)
        for tax_idx, tax_val in enumerate(tax_rate_vals):
            row = [tax_val] + list(ratio_grid[tax_idx, :])
            writer.writerow(row)


def write_panel_b_xlsx(path, gini_mesh, tax_rate_mesh, ratio_grid):
    try:
        import pandas as pd
    except ImportError:
        print('pandas not installed; skipping Panel B XLSX export')
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gini_vals = gini_mesh[0, :]
    tax_rate_vals = tax_rate_mesh[:, 0]

    data = {
        'tax_rate': tax_rate_vals,
    }
    for col_idx, g_val in enumerate(gini_vals):
        data[g_val] = ratio_grid[:, col_idx]

    df = pd.DataFrame(data)
    df.to_excel(path, index=False)


def main(config_path, output_pdf, n_quad_override, csv_export, xlsx_export):
    """
    Main execution function.

    Parameters
    ----------
    config_path : str
        Path to JSON configuration file
    output_pdf : str or None
        Output PDF filename (if None, uses default in timestamped directory)
    n_quad_override : int or None
        Number of quadrature points override
    csv_export : bool
        Whether to export Panel A and Panel B as CSV files
    xlsx_export : bool
        Whether to export Panel A and Panel B as XLSX files
    """
    config = load_configuration(config_path)

    run_name = config.run_name
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = Path('data') / 'output' / f'utility_ratio_plots_{run_name}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_out_path(maybe_path, default_name):
        if maybe_path is None:
            return output_dir / default_name
        candidate = Path(maybe_path)
        return candidate if candidate.is_absolute() else output_dir / candidate

    output_pdf_path = resolve_out_path(output_pdf, f'{run_name}_utility_ratios.pdf')

    params_t0 = evaluate_params_at_time(0.0, config)

    eta = params_t0['eta']
    s = params_t0['s']
    y_net_reference = config.scalar_params.y_net_reference
    use_empirical_lorenz = config.scalar_params.use_empirical_lorenz

    A0 = params_t0['A']
    L0 = params_t0['L']
    K0 = config.scalar_params.K_initial
    alpha = config.scalar_params.alpha
    y_gross = A0 * (K0 / L0) ** alpha

    psi1 = config.scalar_params.psi1
    psi2 = config.scalar_params.psi2
    Omega_base = psi1 * DELTA_T_REPRESENTATIVE + psi2 * (DELTA_T_REPRESENTATIVE ** 2)

    if n_quad_override is not None:
        n_quad = n_quad_override
    else:
        n_quad = config.integration_params.n_quad

    base_params_damage = {
        'y_gross': 1.0,
        'eta': eta,
        's': s,
        'Omega_base': Omega_base,
        'y_net_reference': 1.0,
        'use_empirical_lorenz': use_empirical_lorenz,
    }

    base_params_tax = {
        'y_gross': y_gross,
        'eta': eta,
        's': s,
        'use_empirical_lorenz': use_empirical_lorenz,
    }

    print(f'Representative economic state:')
    print(f'  y_gross = {y_gross:.2f} $/person')
    print(f'  Omega_base = {Omega_base:.6f} (at ΔT = {DELTA_T_REPRESENTATIVE}°C)')
    print(f'  eta = {eta:.3f}')
    print(f'  s = {s:.3f}')
    print(f'  n_quad = {n_quad}')
    print()

    print('Computing Panel A: Climate damage utility ratios...')
    gini_vals_damage = np.linspace(GINI_RANGE_DAMAGE[0], GINI_RANGE_DAMAGE[1], N_GINI_DAMAGE)
    exponent_vals = np.linspace(EXPONENT_RANGE[0], EXPONENT_RANGE[1], N_EXPONENT)
    gini_mesh_damage, exponent_mesh = np.meshgrid(gini_vals_damage, exponent_vals)

    ratio_grid_damage = compute_damage_ratio_grid(gini_mesh_damage, exponent_mesh, base_params_damage, n_quad)

    print(f'  Panel A ratio range: [{np.min(ratio_grid_damage):.4f}, {np.max(ratio_grid_damage):.4f}]')
    print(f'  Panel A ratio at exponent=0, Gini=0: {ratio_grid_damage[0, 0]:.4f}')
    print(f'  Panel A ratio at exponent=0, Gini=0.7: {ratio_grid_damage[0, -1]:.4f}')

    min_idx = np.unravel_index(np.argmin(ratio_grid_damage), ratio_grid_damage.shape)
    min_gini = gini_vals_damage[min_idx[1]]
    min_exp = exponent_vals[min_idx[0]]
    print(f'  Panel A minimum at Gini={min_gini:.3f}, exponent={min_exp:.3f}')

    print('Computing Panel B: Progressive taxation utility ratios...')
    gini_vals_tax = np.linspace(GINI_RANGE_TAX[0], GINI_RANGE_TAX[1], N_GINI_TAX)
    tax_rate_vals = np.linspace(TAX_RATE_RANGE[0], TAX_RATE_RANGE[1], N_TAX_RATE)
    gini_mesh_tax, tax_rate_mesh = np.meshgrid(gini_vals_tax, tax_rate_vals)

    ratio_grid_tax = compute_tax_ratio_grid(gini_mesh_tax, tax_rate_mesh, base_params_tax, n_quad)

    print(f'  Panel B ratio range: [{np.min(ratio_grid_tax):.4f}, {np.max(ratio_grid_tax):.4f}]')
    print(f'  Panel B ratio at tax_rate=0, Gini=0: {ratio_grid_tax[0, 0]:.4f}')
    print(f'  Panel B ratio at tax_rate=0, Gini=0.7: {ratio_grid_tax[0, -1]:.4f}')

    damage_data = {
        'gini_mesh': gini_mesh_damage,
        'exponent_mesh': exponent_mesh,
        'ratio_grid': ratio_grid_damage,
    }

    tax_data = {
        'gini_mesh': gini_mesh_tax,
        'tax_rate_mesh': tax_rate_mesh,
        'ratio_grid': ratio_grid_tax,
    }

    if csv_export:
        panel_a_csv_path = output_dir / f'{run_name}_panel_a.csv'
        panel_b_csv_path = output_dir / f'{run_name}_panel_b.csv'
        write_panel_a_csv(panel_a_csv_path, gini_mesh_damage, exponent_mesh, ratio_grid_damage)
        write_panel_b_csv(panel_b_csv_path, gini_mesh_tax, tax_rate_mesh, ratio_grid_tax)
        print(f'Exported CSV files: {panel_a_csv_path}, {panel_b_csv_path}')

    if xlsx_export:
        panel_a_xlsx_path = output_dir / f'{run_name}_panel_a.xlsx'
        panel_b_xlsx_path = output_dir / f'{run_name}_panel_b.xlsx'
        write_panel_a_xlsx(panel_a_xlsx_path, gini_mesh_damage, exponent_mesh, ratio_grid_damage)
        write_panel_b_xlsx(panel_b_xlsx_path, gini_mesh_tax, tax_rate_mesh, ratio_grid_tax)
        print(f'Exported XLSX files: {panel_a_xlsx_path}, {panel_b_xlsx_path}')

    print(f'Creating 2-panel PDF figure: {output_pdf_path}')
    create_two_panel_figure(damage_data, tax_data, output_pdf_path)

    print(f'Output saved to: {output_dir}')
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate 2-panel PDF of utility ratios under climate damage and taxation scenarios'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to JSON configuration file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output PDF filename (default: {run_name}_utility_ratios.pdf in timestamped directory)'
    )
    parser.add_argument(
        '--n-quad',
        type=int,
        default=None,
        help='Number of quadrature points (default: from config or 32)'
    )

    parser.add_argument(
        '--csv',
        action='store_true',
        help='Export both Panel A and Panel B data as CSV files'
    )
    parser.add_argument(
        '--xlsx',
        action='store_true',
        help='Export both Panel A and Panel B data as XLSX files (requires pandas)'
    )

    args = parser.parse_args()
    main(args.config, args.output, args.n_quad, args.csv, args.xlsx)
