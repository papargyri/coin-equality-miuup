"""
DICE-style miuup (μ upper bound) schedule calculation.

Implements time-varying cap on abatement fraction μ following DICE conventions.
The schedule uses 5-year periods with values defined at period boundaries,
and linear interpolation for yearly resolution.

DICE-2023 schedule at period boundaries:
- p=1 (2020): 0.05
- p=2 (2025): 0.10
- p>=3: delmiumax * (p-1) where delmiumax=0.12
- p>8: override with 0.85 + 0.05*(p-8)

With linear interpolation, yearly values are computed as:
mu_cap(year) = mu0 + w * (mu1 - mu0)
where w = (year - y0) / period_years
"""

from constants import (
    MIUUP_DELMIUMAX,
    MIUUP_P1_VALUE,
    MIUUP_P2_VALUE,
    MIUUP_PLATEAU_START_PERIOD,
    MIUUP_PLATEAU_BASE,
    MIUUP_PLATEAU_INCREMENT,
)


def get_period_from_year(year, start_year, period_years):
    """
    Map calendar year to DICE period (1-indexed).

    Parameters
    ----------
    year : int or float
        Calendar year (e.g., 2020, 2021, ...)
    start_year : int
        First year of period 1 (default 2020)
    period_years : int
        Length of each period in years (default 5)

    Returns
    -------
    int
        Period index (1-indexed): p=1 for years 2020-2024, p=2 for 2025-2029, etc.
    """
    return 1 + int((year - start_year) // period_years)


def get_miuup_at_period_boundary(period):
    """
    Get μ_up value at the START of a DICE period (at the 5-year boundary).

    DICE-2023 schedule (EXACT as specified):
    - p=1: 0.05
    - p=2: 0.10
    - p>=3: delmiumax * (p-1) where delmiumax=0.12
    - OVERRIDE if p>8: 0.85 + 0.05*(p-8)

    Parameters
    ----------
    period : int
        Period index (1-indexed)

    Returns
    -------
    float
        μ_up value at the start of this period
    """
    if period <= 1:
        mu = MIUUP_P1_VALUE  # 0.05
    elif period == 2:
        mu = MIUUP_P2_VALUE  # 0.10
    else:
        # p >= 3: use delmiumax formula
        mu = MIUUP_DELMIUMAX * (period - 1)

    # Override for p > 8 (plateau formula)
    if period > MIUUP_PLATEAU_START_PERIOD:
        mu = MIUUP_PLATEAU_BASE + MIUUP_PLATEAU_INCREMENT * (period - MIUUP_PLATEAU_START_PERIOD)

    return mu


def get_miuup_at_year(year, start_year, period_years, interpolation="linear"):
    """
    Get μ_up value for a specific calendar year.

    For "linear" interpolation, computes value by linear interpolation
    between adjacent 5-year period boundaries.

    Parameters
    ----------
    year : int or float
        Calendar year (e.g., 2020, 2021, ...)
    start_year : int
        First year of period 1 (default 2020)
    period_years : int
        Length of each period in years (default 5)
    interpolation : str
        Interpolation method: "linear" or "stepwise"

    Returns
    -------
    float
        μ_up value for this year

    Examples
    --------
    >>> get_miuup_at_year(2020, 2020, 5)
    0.05
    >>> get_miuup_at_year(2021, 2020, 5)
    0.06
    >>> get_miuup_at_year(2025, 2020, 5)
    0.10
    """
    # Step 1: compute period index
    period = get_period_from_year(year, start_year, period_years)

    # Step 2: compute start year of this period (y0)
    y0 = start_year + (period - 1) * period_years

    if interpolation == "stepwise":
        # Stepwise: use period boundary value directly
        return get_miuup_at_period_boundary(period)

    # Linear interpolation
    # Step 3: get mu values at adjacent boundaries
    mu0 = get_miuup_at_period_boundary(period)
    mu1 = get_miuup_at_period_boundary(period + 1)

    # Step 4: compute interpolation weight
    # w = 0 at y0, w = 1 at y0 + period_years
    w = (year - y0) / period_years

    # Step 5: linear interpolation
    return mu0 + w * (mu1 - mu0)


def invert_abatement_cost(mu, epot, theta1, theta2):
    """
    Compute abatement cost from μ by inverting Eq 1.6.

    Given Eq 1.6:
        μ = [AbateCost · θ₂ / (E_pot · θ₁)]^(1/θ₂)

    Invert to get:
        AbateCost = (E_pot · θ₁ / θ₂) · μ^θ₂

    Parameters
    ----------
    mu : float
        Abatement fraction (capped)
    epot : float
        Potential emissions per capita (tCO2/person)
    theta1 : float
        Abatement cost coefficient ($/tCO2)
    theta2 : float
        Abatement cost exponent

    Returns
    -------
    float
        Abatement cost per capita corresponding to this μ
    """
    return (epot * theta1 / theta2) * (mu ** theta2)


def print_miuup_schedule(start_year, end_year, period_years, interpolation="linear"):
    """
    Print μ_up schedule for verification.

    Parameters
    ----------
    start_year : int
        First year to print
    end_year : int
        Last year to print (inclusive)
    period_years : int
        Length of each DICE period in years
    interpolation : str
        Interpolation method
    """
    print(f"\nMiuup Schedule (interpolation={interpolation}):")
    print(f"{'Year':<6} {'mu_cap':>8}")
    print("-" * 16)
    for year in range(start_year, end_year + 1):
        mu_cap = get_miuup_at_year(year, start_year, period_years, interpolation)
        print(f"{year:<6} {mu_cap:>8.4f}")
    print()


if __name__ == "__main__":
    # Validation: print schedule for years 2020-2031
    print("=" * 50)
    print("MIUUP SCHEDULE VALIDATION")
    print("=" * 50)

    print("\nExpected values:")
    print("2020: 0.05")
    print("2021: 0.06")
    print("2022: 0.07")
    print("2023: 0.08")
    print("2024: 0.09")
    print("2025: 0.10")

    print_miuup_schedule(2020, 2031, 5, "linear")

    # Also show period boundary values
    print("\nPeriod boundary values (for reference):")
    print(f"{'Period':<8} {'Year':<6} {'mu_cap':>8}")
    print("-" * 24)
    for p in range(1, 15):
        year = 2020 + (p - 1) * 5
        mu = get_miuup_at_period_boundary(p)
        print(f"p={p:<5} {year:<6} {mu:>8.4f}")
