
\textit{Can you explain to me how social cost of carbon, scc, is calculated in this code?}

In this DICE GAMS code, **SCC is computed *after* the nonlinear program is solved**, using the **dual values (marginals / Lagrange multipliers) of two equations**.

You can see the actual line in both:

* `/mnt/data/def-opt-b-4-3-10.gms`
* `/mnt/data/def-base-b-4-3-1.gms`

```gams
scc(t) = -1000*eco2eq.m(t)/(.00001+cc.m(t));
```

## What that means

### 1) The model is solved as a welfare maximization problem

Both files do something like:

```gams
solve CO2 maximizing UTILITY using nlp ;
```

So GAMS chooses the control path (e.g., **MIU(t)** abatement, savings, etc.) to maximize **UTILITY** subject to the model equations.

### 2) `eco2eq` is the “industrial emissions” identity, and `eco2eq.m(t)` is its shadow value

In the main model file (`DICE2023-b-4-3-10.gms`) the emissions equation is:

```gams
eco2eq(t).. ECO2(t) =E= (sigma(t)*YGROSS(t) + eland(t))*(1-(MIU(t)));
```

* `ECO2(t)` is emissions (in DICE, effectively **GtCO₂ per year** in each period).
* `.m` in GAMS is the **marginal/dual** on that equation at the solution.

Intuitively, `eco2eq.m(t)` captures how the objective would change if you “relaxed” that emissions relationship by 1 unit of emissions (i.e., allow a tiny extra unit of `ECO2(t)` holding the rest of the structure consistent). Because extra emissions are bad for welfare, this shadow value is typically **negative**.

### 3) `cc` is the consumption identity, and `cc.m(t)` is the marginal value of consumption

Consumption is defined by:

```gams
cc(t).. C(t) =E= Y(t) - I(t);
```

So `cc.m(t)` is (up to sign conventions) the **shadow value of one more unit of consumption** in period *t* in the welfare objective — i.e., the model’s **marginal utility value of consumption in “objective units.”**

### 4) SCC is the ratio: “marginal welfare impact of emissions” divided by “marginal welfare value of consumption”

That line is implementing the standard definition:

[
\text{SCC}_t ;=; - \frac{\partial W/\partial E_t}{\partial W/\partial C_t}
]

In the code’s terms:

* (\partial W/\partial E_t) is represented by `eco2eq.m(t)`
* (\partial W/\partial C_t) is represented by `cc.m(t)`

So:

```gams
scc(t) ∝ - eco2eq.m(t) / cc.m(t)
```

The **minus sign** is there because extra emissions reduce welfare (negative numerator) but SCC is reported as a **positive cost**.


---

## Bottom line

**In this code, SCC(t) is not computed by explicitly perturbing emissions and re-running the model.** It is computed from the **first-order conditions of the solved NLP** as:

* the **shadow value of emissions** (`eco2eq.m(t)`)
* divided by the **shadow value of consumption** (`cc.m(t)`)
* with a sign flip and a units conversion to **$/tCO₂**.

If you want, I can also point out (in the main `.gms`) where these equations enter the full Lagrangian logic and how this relates to the “carbon price / shadow price of carbon” concept elsewhere in DICE (sometimes there’s also a carbon price variable/equation like `CPRICE`, but SCC here is coming straight from duals).


\textis{How might the constraint provided by miuup affect the dual values used to calculated ssc?}

Short answer: **`miuup` can materially distort the duals used in the SCC formula, especially when it binds**, because the SCC is being inferred from *local shadow values*, and a binding control constraint changes those shadow values away from the unconstrained “Pigouvian” ones.

Here’s the structured explanation.

---

## 1. What `miuup` is doing conceptually

In DICE, `MIU(t)` is the emissions control rate (abatement fraction).
A constraint like

```gams
MIU(t) =L= miuup(t)
```

imposes an **upper bound on abatement**.

Economically, this says:

> “Even if it would be welfare-improving to abate more, you are not allowed to.”

This is a *real* constraint (technology, politics, inertia), but it is **not part of the social planner’s first-best problem**.

---

## 2. Reminder: how SCC is computed in this code

The SCC is computed as

[
\text{SCC}_t = - \frac{\lambda^{E}_t}{\lambda^{C}_t}
]

where:

* (\lambda^{E}_t = \texttt{eco2eq.m(t)})
  → marginal welfare impact of emissions
* (\lambda^{C}_t = \texttt{cc.m(t)})
  → marginal welfare value of consumption

These are **Lagrange multipliers of equilibrium constraints**, not structural damage parameters.

---

## 3. What happens when `miuup` is *not binding*

If `MIU(t) < miuup(t)` at the solution:

* The `miuup` constraint has **zero shadow value**
* The FOCs with respect to `MIU(t)` are the usual interior ones
* Emissions are at their welfare-optimal level

In this case:

* `eco2eq.m(t)` reflects the *true marginal damage of emissions*
* SCC corresponds to the standard Pigouvian SCC

➡️ **No distortion** from `miuup`.

---

## 4. What happens when `miuup` *binds*

If `MIU(t) = miuup(t)`:

* The constraint has a **positive shadow price**
* The planner *wants* more abatement but is prevented
* Emissions are **too high relative to the unconstrained optimum**

This changes the duals in two important ways.

---

## 5. Effect on `eco2eq.m(t)` (numerator of SCC)

When `miuup` binds:

* The marginal damage of emissions is **partially absorbed by the binding MIU constraint**
* The Lagrange multiplier on `eco2eq` no longer reflects the *full* welfare cost of emissions
* Intuitively:

  > “Extra emissions are bad, but since I can’t abate more anyway, their marginal value is muted.”

Formally:

* The emissions shadow value is now a **second-best shadow price**
* It is conditional on the abatement ceiling

👉 **`eco2eq.m(t)` is biased toward zero (in absolute value)** relative to the first-best case.

---

## 6. Effect on `cc.m(t)` (denominator)

The marginal utility of consumption typically changes much less:

* `cc.m(t)` mainly reflects CRRA utility and discounting
* It is only indirectly affected via lower output and consumption levels

So the dominant distortion comes from the **numerator**, not the denominator.

---

## 7. Net effect on reported SCC

Because:

* `eco2eq.m(t)` is **too small in magnitude**
* `cc.m(t)` is roughly unchanged

the ratio

[
\text{SCC}_t = -\frac{\lambda^E_t}{\lambda^C_t}
]

is typically:

➡️ **biased downward when `miuup` binds**

That is:

* Reported SCC < true marginal damage
* SCC becomes a **policy-constrained shadow price**, not a structural damage metric

---

## 8. Interpretation: what SCC means when `miuup` binds

When `miuup` is binding, the SCC in this code should be interpreted as:

> “The marginal value of reducing emissions *given that further abatement is infeasible*.”

It is **not**:

* The welfare loss of an extra ton of CO₂ in a first-best world
* The correct Pigouvian tax for a world without the abatement cap

This is a classic second-best result.

---

## 9. Practical implications for using SCC from this run

If your goal is:

### ✔️ Policy analysis *with* feasibility constraints

Then this SCC is internally consistent and meaningful.

### ❌ Estimating a “pure” SCC for comparison or reporting

Then you should:

* Ensure `miuup` is slack in the periods used
* Or compute SCC via a perturbation experiment
* Or solve a relaxed model (temporarily remove or loosen `miuup`) and extract SCC there

---

## 10. One-sentence takeaway

**A binding `miuup` constraint causes the SCC computed from duals to become a second-best, downward-biased shadow price that reflects abatement infeasibility rather than the true marginal damage of emissions.**

If you want, I can also:

* Show how this appears explicitly in the MIU FOCs
* Sketch a clean “perturb emissions by ε” SCC calculation for comparison
* Help design a diagnostic to flag periods where SCC is contaminated by binding MIU constraints
