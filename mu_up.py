"""
Mu_up (μ upper bound) schedule calculation.

Implements time-varying cap on abatement fraction μ using user-defined schedules.
The schedule is specified as a list of [year, mu_cap] pairs with linear
interpolation between points and flat extrapolation outside the range.

Example schedule:
[[2020, 0.05], [2025, 0.10], [2060, 0.9], [2070, 1.0]]

For years before 2020: mu_cap = 0.05 (first value)
For years after 2070: mu_cap = 1.0 (last value)
For years between points: linear interpolation
"""

import numpy as np


def get_mu_up_from_schedule(year, schedule):
    """
    Get μ_up value for a specific calendar year from user-defined schedule.

    Uses linear interpolation between schedule points. For times before the
    first point or after the last point, returns the value at that boundary
    (flat extrapolation).

    Parameters
    ----------
    year : float
        Calendar year (e.g., 2020.0, 2025.5, ...)
    schedule : list of [year, mu_cap] pairs
        User-defined schedule. Must be sorted by year (ascending).
        Example: [[2020, 0.05], [2025, 0.10], [2060, 0.9], [2070, 1.0]]

    Returns
    -------
    float
        μ_up value for this year

    Examples
    --------
    >>> schedule = [[2020, 0.05], [2025, 0.10], [2070, 1.0]]
    >>> get_mu_up_from_schedule(2020, schedule)
    0.05
    >>> get_mu_up_from_schedule(2022.5, schedule)
    0.075
    >>> get_mu_up_from_schedule(2010, schedule)  # Before first point
    0.05
    >>> get_mu_up_from_schedule(2100, schedule)  # After last point
    1.0
    """
    # Extract years and values from schedule
    years = np.array([point[0] for point in schedule])
    values = np.array([point[1] for point in schedule])

    # np.interp handles flat extrapolation by default when year is outside range
    return np.interp(year, years, values)


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


def print_mu_up_schedule(schedule, start_year, end_year):
    """
    Print μ_up schedule for verification.

    Parameters
    ----------
    schedule : list of [year, mu_cap] pairs
        User-defined schedule
    start_year : int
        First year to print
    end_year : int
        Last year to print (inclusive)
    """
    print(f"\nMu_up Schedule:")
    print(f"{'Year':<6} {'mu_cap':>8}")
    print("-" * 16)
    for year in range(start_year, end_year + 1):
        mu_cap = get_mu_up_from_schedule(year, schedule)
        print(f"{year:<6} {mu_cap:>8.4f}")
    print()


if __name__ == "__main__":
    # Validation: print schedule using example user-defined schedule
    print("=" * 50)
    print("MU_UP SCHEDULE VALIDATION")
    print("=" * 50)

    # Example schedule matching DICE-2023 early years
    example_schedule = [
        [2020, 0.05],
        [2025, 0.10],
        [2030, 0.24],
        [2060, 0.9],
        [2070, 1.0],
    ]

    print("\nExample schedule points:")
    for year, value in example_schedule:
        print(f"  {year}: {value}")

    print_mu_up_schedule(example_schedule, 2020, 2080)
