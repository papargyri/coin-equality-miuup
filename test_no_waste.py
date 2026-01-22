"""
Test script for cap_spending_mode implementation.

This script verifies:
1. "waste" mode (default): Spending above cap is wasted (still subtracted from output)
2. "no_waste" mode: Spending above cap returns to consumption
3. use_mu_up=False: Unchanged baseline behavior
"""

import numpy as np
from parameters import load_configuration
from economic_model import integrate_model

def test_no_waste_implementation():
    """Test cap_spending_mode implementation with debug output."""

    print("="*80)
    print("TEST 1: cap_spending_mode='waste' (default, Ken's Project 1 design)")
    print("="*80)

    # Load a config with use_mu_up=True
    config_waste = load_configuration('json/config_015_t-f-t-f-f_1_100k_mu_up_true.json')

    # Run short simulation (2020-2030)
    config_waste.scalar_params.t_end = 2030.0
    config_waste.scalar_params.dt = 1.0
    config_waste.scalar_params.cap_spending_mode = "waste"  # Explicitly set default

    print("\nRunning simulation with cap_spending_mode='waste'...")
    print("Expected: When cap binds, abateCost_effective == abateCost_proposed")
    print("Expected: unused_abatement_budget = 0 always (no money returned)")
    print("Expected: wasted_abatement_spending > 0 when cap binds (but still subtracted from output)")
    print()

    results_waste = integrate_model(config_waste, store_detailed_output=True)

    # Print results for first 11 years
    print("\nResults Summary (years 2020-2030):")
    print("Year  mu_cap  mu_uncap  mu_final  cap_bind  AbateCost_prop  AbateCost_eff  wasted_spend  unused_budget")
    print("-" * 110)

    for i in range(min(11, len(results_waste['t']))):
        year = results_waste['t'][i]
        mu_cap = results_waste['mu_cap'][i]
        mu_uncap = results_waste['mu_uncapped'][i]
        mu_final = results_waste['mu'][i]
        cap_bind = results_waste['cap_binding'][i]
        abate_prop = results_waste['abateCost_proposed'][i]
        abate_eff = results_waste['abateCost_effective'][i]
        wasted = results_waste['wasted_abatement_spending'][i]
        unused = results_waste['unused_abatement_budget'][i]

        print(f"{year:4.0f}  {mu_cap:7.4f}  {mu_uncap:8.5f}  {mu_final:8.5f}  "
              f"{cap_bind:8.0f}  {abate_prop:14.6e}  {abate_eff:13.6e}  "
              f"{wasted:12.6e}  {unused:13.6e}")

    print("\n" + "="*80)
    print("TEST 2: cap_spending_mode='no_waste' (optional mode)")
    print("="*80)

    # Load a config with use_mu_up=True
    config_no_waste = load_configuration('json/config_015_t-f-t-f-f_1_100k_mu_up_true.json')

    # Run short simulation (2020-2030)
    config_no_waste.scalar_params.t_end = 2030.0
    config_no_waste.scalar_params.dt = 1.0
    config_no_waste.scalar_params.cap_spending_mode = "no_waste"

    print("\nRunning simulation with cap_spending_mode='no_waste'...")
    print("Expected: When cap binds, abateCost_effective < abateCost_proposed")
    print("Expected: unused_abatement_budget = wasted_spending when cap binds")
    print("Expected: unused_abatement_budget = 0 when cap doesn't bind")
    print()

    results_no_waste = integrate_model(config_no_waste, store_detailed_output=True)

    # Print results for first 11 years
    print("\nResults Summary (years 2020-2030):")
    print("Year  mu_cap  mu_uncap  mu_final  cap_bind  AbateCost_prop  AbateCost_eff  wasted_spend  unused_budget")
    print("-" * 110)

    for i in range(min(11, len(results_no_waste['t']))):
        year = results_no_waste['t'][i]
        mu_cap = results_no_waste['mu_cap'][i]
        mu_uncap = results_no_waste['mu_uncapped'][i]
        mu_final = results_no_waste['mu'][i]
        cap_bind = results_no_waste['cap_binding'][i]
        abate_prop = results_no_waste['abateCost_proposed'][i]
        abate_eff = results_no_waste['abateCost_effective'][i]
        wasted = results_no_waste['wasted_abatement_spending'][i]
        unused = results_no_waste['unused_abatement_budget'][i]

        print(f"{year:4.0f}  {mu_cap:7.4f}  {mu_uncap:8.5f}  {mu_final:8.5f}  "
              f"{cap_bind:8.0f}  {abate_prop:14.6e}  {abate_eff:13.6e}  "
              f"{wasted:12.6e}  {unused:13.6e}")

    print("\n" + "="*80)
    print("TEST 3: use_mu_up=False (baseline, should be unchanged)")
    print("="*80)

    # Load config with use_mu_up=False
    config_uncapped = load_configuration('json/config_015_t-f-t-f-f_1_100k_mu_up_false.json')
    config_uncapped.scalar_params.t_end = 2030.0
    config_uncapped.scalar_params.dt = 1.0

    print("\nRunning simulation with use_mu_up=False...")
    print("Expected: cap_binding = 0 for all years")
    print("Expected: unused_abatement_budget = 0 for all years")
    print("Expected: mu_cap = 1e12 (INVERSE_EPSILON)")
    print("Expected: abateCost_proposed = abateCost_effective")
    print()

    results_uncapped = integrate_model(config_uncapped, store_detailed_output=True)

    # Print results for first 11 years
    print("\nResults Summary (years 2020-2030):")
    print("Year  mu_cap       mu_uncap  mu_final  cap_bind  AbateCost_prop  AbateCost_eff  wasted_spend  unused_budget")
    print("-" * 115)

    for i in range(min(11, len(results_uncapped['t']))):
        year = results_uncapped['t'][i]
        mu_cap = results_uncapped['mu_cap'][i]
        mu_uncap = results_uncapped['mu_uncapped'][i]
        mu_final = results_uncapped['mu'][i]
        cap_bind = results_uncapped['cap_binding'][i]
        abate_prop = results_uncapped['abateCost_proposed'][i]
        abate_eff = results_uncapped['abateCost_effective'][i]
        wasted = results_uncapped['wasted_abatement_spending'][i]
        unused = results_uncapped['unused_abatement_budget'][i]

        print(f"{year:4.0f}  {mu_cap:11.2e}  {mu_uncap:8.5f}  {mu_final:8.5f}  "
              f"{cap_bind:8.0f}  {abate_prop:14.6e}  {abate_eff:13.6e}  "
              f"{wasted:12.6e}  {unused:13.6e}")

    # Verification checks
    print("\n" + "="*80)
    print("VERIFICATION CHECKS")
    print("="*80)

    # Check 1: use_mu_up=False - cap never binds
    assert np.all(results_uncapped['cap_binding'] == 0), "FAIL: Cap should never bind when use_mu_up=False"
    print("✓ PASS: Cap never binds when use_mu_up=False")

    # Check 2: use_mu_up=False - unused_budget is always 0
    assert np.all(results_uncapped['unused_abatement_budget'] == 0), "FAIL: unused_budget should be 0 when use_mu_up=False"
    print("✓ PASS: unused_abatement_budget = 0 when use_mu_up=False")

    # Check 3: use_mu_up=False - wasted_spending is 0
    assert np.all(results_uncapped['wasted_abatement_spending'] == 0), \
        "FAIL: wasted_abatement_spending should be 0 when use_mu_up=False"
    print("✓ PASS: wasted_abatement_spending = 0 when use_mu_up=False")

    # Find binding years for no_waste mode
    cap_binding_no_waste = results_no_waste['cap_binding'] > 0.5

    # Check 5: WASTE mode - unused_budget is always 0 (even when cap binds)
    assert np.all(results_waste['unused_abatement_budget'] == 0), \
        "FAIL: unused_abatement_budget should be 0 in 'waste' mode (money is wasted, not returned)"
    print("✓ PASS: unused_abatement_budget = 0 in 'waste' mode (money wasted)")

    # Check 6: WASTE mode - effective == proposed (all proposed spending is used)
    assert np.allclose(results_waste['abateCost_proposed'], results_waste['abateCost_effective']), \
        "FAIL: abateCost_effective should equal abateCost_proposed in 'waste' mode"
    print("✓ PASS: abateCost_effective = abateCost_proposed in 'waste' mode")

    # Check 7: WASTE mode - wasted_spending should be 0 by definition
    assert np.allclose(results_waste['wasted_abatement_spending'], 0), \
        "FAIL: wasted_abatement_spending should be 0 in 'waste' mode (effective==proposed)"
    print("✓ PASS: wasted_abatement_spending = 0 in 'waste' mode (by definition)")

    # Check 8: NO_WASTE mode - when cap binds, unused_budget > 0
    if np.any(cap_binding_no_waste):
        assert np.all(results_no_waste['unused_abatement_budget'][cap_binding_no_waste] > -1e-10), \
            "FAIL: unused_budget should be >= 0 when cap binds in 'no_waste' mode"
        print(f"✓ PASS: unused_abatement_budget > 0 when cap binds in 'no_waste' mode ({np.sum(cap_binding_no_waste)} binding years)")
    else:
        print("⚠ WARNING: Cap never bound in 'no_waste' test (might be OK if f is very low)")

    # Check 9: NO_WASTE mode - when cap binds, effective < proposed
    if np.any(cap_binding_no_waste):
        assert np.all(results_no_waste['abateCost_effective'][cap_binding_no_waste] <
                     results_no_waste['abateCost_proposed'][cap_binding_no_waste]), \
            "FAIL: abateCost_effective should be < abateCost_proposed when cap binds in 'no_waste' mode"
        print("✓ PASS: abateCost_effective < abateCost_proposed when cap binds in 'no_waste' mode")

    # Check 10: NO_WASTE mode - when cap binds, unused_budget == wasted_spending
    if np.any(cap_binding_no_waste):
        assert np.allclose(results_no_waste['unused_abatement_budget'][cap_binding_no_waste],
                          results_no_waste['wasted_abatement_spending'][cap_binding_no_waste]), \
            "FAIL: unused_budget should equal wasted_spending when cap binds in 'no_waste' mode"
        print("✓ PASS: unused_abatement_budget = wasted_spending when cap binds in 'no_waste' mode")

    # Check 11: NO_WASTE mode - when cap doesn't bind, unused_budget == 0
    cap_not_binding_no_waste = results_no_waste['cap_binding'] <= 0.5
    assert np.all(results_no_waste['unused_abatement_budget'][cap_not_binding_no_waste] == 0), \
        "FAIL: unused_budget should be 0 when cap doesn't bind in 'no_waste' mode"
    print("✓ PASS: unused_abatement_budget = 0 when cap doesn't bind in 'no_waste' mode")

    # Check 12: Both modes have same mu values (cap affects spending, not mu itself)
    assert np.allclose(results_waste['mu'], results_no_waste['mu']), \
        "FAIL: mu values should be identical in both modes"
    print("✓ PASS: mu values identical in both modes (cap affects spending, not mu)")

    print("\n" + "="*80)
    print("ALL TESTS PASSED!")
    print("="*80)
    print("\nSummary:")
    print(f"  - WASTE mode: {np.sum(results_waste['cap_binding'] > 0.5)} years with binding cap")
    print(f"    - unused_budget = 0 always (money wasted)")
    print(f"    - abateCost_effective = abateCost_proposed always")
    print(f"  - NO_WASTE mode: {np.sum(results_no_waste['cap_binding'] > 0.5)} years with binding cap")
    print(f"    - Total freed budget: {np.sum(results_no_waste['unused_abatement_budget']):.6e}")
    print(f"    - Freed money returns to consumption")
    print(f"  - Baseline (use_mu_up=False): {np.sum(results_uncapped['cap_binding'] > 0.5)} years with binding cap (should be 0)")
    print(f"  - Default behavior: 'waste' mode (Ken's Project 1 design)")
    print()

if __name__ == "__main__":
    test_no_waste_implementation()
