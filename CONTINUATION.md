# Continuation: Planned Future Work

## 1. Pass mu_max via schedule (like mu_up_schedule)

Currently `mu_max` is a scalar parameter in `ScalarParameters`. We need to:

1. Allow `mu_max` to be specified as a time-varying schedule (list of `[year, value]` pairs), similar to `mu_up_schedule`
2. Pass the interpolated `mu_max` value from `integrate_model()` to `calculate_tendencies()`
3. If no schedule is provided, default to `INVERSE_EPSILON` (effectively no cap)

### Implementation approach:
- Add `mu_max_schedule` field to `ScalarParameters` (optional, list of `[year, mu_max]` pairs)
- In `integrate_model()`, interpolate `mu_max` for each timestep using linear interpolation (same as `mu_up_schedule`)
- Pass interpolated value to `calculate_tendencies()` via `mu_up_params` dict (or a new dict)
- Fall back to `INVERSE_EPSILON` if `mu_max_schedule` is None

## 2. Add CO2 emissions at specified points in time

Allow injection of additional CO2 emissions at specified times, independent of economic activity.

### Use cases:
- Modeling volcanic eruptions or other natural CO2 sources
- Scenario analysis with pulse emissions
- Historical emissions not captured by the economic model

### Implementation approach:
- Add `co2_injections` field to configuration: list of `[year, amount_tCO2]` pairs
- In `integrate_model()`, add injected emissions to `Ecum` at the appropriate timesteps
- Could be instantaneous pulses or spread over time intervals

## 3. Add money available for consumption at specified points in time

Allow injection of additional consumption capacity at specified times.

### Use cases:
- Modeling windfall gains (e.g., resource discoveries)
- External aid or transfers
- Scenario analysis with consumption shocks

### Implementation approach:
- Add `consumption_injections` field to configuration: list of `[year, amount_dollars_per_capita]` pairs
- In `calculate_tendencies()`, add injected amount to consumption before utility calculation
- Linear interpolation between specified points, or instantaneous pulses
