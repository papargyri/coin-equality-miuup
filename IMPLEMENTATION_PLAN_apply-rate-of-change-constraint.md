# Implementation Plan: Constrain Rate of Change of Control Variable f

## Goal
Implement a constraint limiting how fast the control variable f can change per year:
```
f0 - (t1 - t0) * max_rate <= f1 <= f0 + (t1 - t0) * max_rate
```

## Implementation Approaches Considered

### Approach 1: Variable Transformation (Recommended)
**Concept**: Remap optimization variable o ∈ [0,1] to constrained f range based on previous f value.

**Implementation**:
```python
# For first point: f[0] = o[0] (no previous value to constrain against)
f[0] = o[0]

# For subsequent points:
for i in range(1, n_points):
    dt = t[i] - t[i-1]

    # Compute allowed range based on previous f and rate limit
    f_min = max(0.0, f[i-1] - dt * max_rate)
    f_max = min(1.0, f[i-1] + dt * max_rate)

    # Map o[i] ∈ [0,1] to f[i] ∈ [f_min, f_max]
    f[i] = f_min + o[i] * (f_max - f_min)
```

**How it works:**
- When `o[i] = 0`: f takes the minimum allowed value (maximum decrease from f[i-1])
- When `o[i] = 1`: f takes the maximum allowed value (maximum increase from f[i-1])
- When `o[i] = 0.5`: f stays approximately at f[i-1]

**Pros**:
- Constraints satisfied by construction - no feasibility issues
- Works with current LN_SBPLX algorithm (no algorithm change needed)
- No penalty tuning required

**Cons**:
- Changes the optimization landscape (gradients are transformed)
- Requires special handling for f[0] (no previous value)
- Implementation complexity in the objective function

### Approach 2: Penalty Function
**Concept**: Return large negative value if constraints violated.

**Pros**:
- Simple to implement
- Works with any algorithm

**Cons**:
- Creates discontinuities in objective landscape
- May cause optimizer to struggle near boundaries
- Need to choose penalty magnitude

### Approach 3: NLopt Inequality Constraints
**Concept**: Use NLopt's built-in constraint mechanism.

**Implementation**:
```python
# For each adjacent pair (i-1, i):
opt.add_inequality_constraint(lambda x, grad: x[i] - x[i-1] - dt*max_rate, 1e-8)
opt.add_inequality_constraint(lambda x, grad: x[i-1] - x[i] - dt*max_rate, 1e-8)
```

**Pros**:
- Clean separation of objective and constraints
- NLopt handles constraint satisfaction
- Mathematically elegant

**Cons**:
- Only works with certain algorithms: COBYLA (LN_COBYLA), SLSQP (LD_SLSQP), MMA (LD_MMA), ISRES (GN_ISRES)
- Current default LN_SBPLX does NOT support inequality constraints
- Would require algorithm change

### Approach 4: Reparameterization as Increments
**Concept**: Optimize changes Δf[i] instead of absolute values, then reconstruct by cumulative sum.

**Implementation**:
```python
# Optimization variables: [f0, Δf1, Δf2, ..., ΔfN]
# Bounds: f0 ∈ [0,1], Δf[i] ∈ [-dt*max_rate, +dt*max_rate]
# Reconstruction: f[i] = f[0] + sum(Δf[1:i+1])
# Must also enforce f[i] ∈ [0,1] after reconstruction
```

**Pros**:
- Rate constraint naturally embedded in bounds
- Works with any algorithm

**Cons**:
- f[i] could still violate [0,1] bounds after reconstruction
- Need additional clipping or penalty for absolute bounds
- Changes interpretation of optimization variables

### Approach 5: Barrier Function
**Concept**: Add smooth penalty that increases near constraint boundaries.

**Implementation**:
```python
barrier = -mu * sum(log(f[i] - f[i-1] + dt*max_rate) + log(f[i-1] - f[i] + dt*max_rate))
objective = utility_integral + barrier
```

**Pros**:
- Smooth objective (good for gradient-based methods)
- Solutions stay interior to feasible region

**Cons**:
- Requires tuning barrier parameter mu
- Never exactly satisfies constraints (only approaches them)
- Can cause numerical issues near boundaries

### Approach 6: Projection Method
**Concept**: Optimize freely, then project solution onto feasible set.

**Implementation**: After optimization, enforce constraints by sequential adjustment:
```python
for i in range(1, n):
    f[i] = clip(f[i], f[i-1] - dt*max_rate, f[i-1] + dt*max_rate)
```

**Pros**:
- Simple post-processing
- Works with any algorithm

**Cons**:
- Projected solution may not be optimal
- Optimizer unaware of constraints during search
- May need to iterate (optimize → project → optimize...)

## Recommendation

**Primary recommendation: Approach 1 (Variable Transformation)**

Reasons:
1. Constraints satisfied by construction - no feasibility issues
2. Works with current LN_SBPLX algorithm (no algorithm change needed)
3. No penalty tuning required
4. Consistent with how bounds are already handled

**Alternative: Approach 3 (NLopt Constraints) with algorithm change to LN_COBYLA**

If cleaner separation of concerns is preferred, switching to COBYLA which supports inequality constraints natively.

## Configuration Changes Needed

Add to `optimization_parameters` in JSON config:
```json
"max_rate_f": 0.05,
"_max_rate_f": "Maximum allowed change in f per year. If null or absent, no rate constraint applied."
```

## Files to Modify

1. **parameters.py**: Add `max_rate_f` to `OptimizationParameters` dataclass
2. **optimization.py**: Implement the constraint mechanism in:
   - `optimize_control_points()` (single variable)
   - `optimize_control_points_f_and_s()` (dual variable)
   - `calculate_objective()` or a new transformation function
3. **json/*.json**: Add parameter to config files (optional, can default to None)

## Open Questions

1. What should happen at iteration boundaries when control points are interpolated from previous solution? (The interpolated values should already approximately satisfy constraints if previous solution did)
2. What is a reasonable default value, or should it default to None (unconstrained)?
3. Should f[0] be constrained relative to some initial condition, or free to take any value in [0,1]?
