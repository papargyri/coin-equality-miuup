# COIN_equality

A simple-as-possible stylized representation of the tradeoff between investment in income redistribution versus investment in emissions abatement.

## Overview

This project develops a highly stylized model of an economy with income inequality, where a specified fraction of gross production is allocated to social good. The central question is how to optimally allocate resources between two competing objectives:

1. **Income redistribution** - reducing inequality by transferring income from high-income to low-income individuals
2. **Emissions abatement** - reducing carbon emissions to mitigate future climate damage

The model extends the COIN framework presented in [Caldeira et al. (2023)](https://doi.org/10.1088/1748-9326/acf949) to incorporate income inequality and diminishing marginal utility of income.

For detailed technical documentation, see [README_DETAIL.md](README_DETAIL.md).

## Installation and Requirements

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installing Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

Required packages include:
- numpy - numerical computations
- scipy - scientific computing and optimization
- matplotlib - plotting and visualization
- nlopt - nonlinear optimization
- openpyxl - Excel file generation

## Quick Start

### Running a Basic Simulation

Test the model with the baseline configuration:

```bash
python test_integration.py config_baseline.json
```

This will:
1. Load the baseline configuration
2. Run the forward integration over the specified time period
3. Display results summary
4. Generate timestamped output directory with CSV data and PDF plots

### Running an Optimization

Optimize the allocation between redistribution and abatement:

```bash
python run_optimization.py config_baseline.json
```

This will find the optimal time trajectory of the control variable f(t) (fraction allocated to abatement vs. redistribution) that maximizes the discounted time-integral of aggregate utility.

### Comparing Multiple Scenarios

After running multiple optimizations:

```bash
python compare_results.py "data/output/scenario_*/"
```

This generates Excel spreadsheets and PDF plots comparing results across different runs.

## Model Overview

### Objective Function

The model optimizes the time-integral of aggregate utility by choosing the allocation fraction f(t) between emissions abatement and income redistribution:

```
max∫₀^∞ e^(-ρt) · U(t) · L(t) dt,  subject to 0 ≤ f(t) ≤ 1
```

where:
- ρ = pure rate of time preference
- U(t) = mean utility of the population at time t
- L(t) = population at time t
- f(t) = fraction of resources allocated to abatement (control variable)

### Core Components

The model combines three subsystems:

1. **Economic Model (Solow-Swan Growth)**
   - Cobb-Douglas production function
   - Capital accumulation with depreciation
   - Climate damage reducing output
   - Income distribution (Pareto-Lorenz)

2. **Climate Model**
   - Temperature proportional to cumulative emissions
   - Emissions from economic activity
   - Abatement reducing emissions

3. **Utility and Inequality**
   - CRRA (isoelastic) utility function
   - Income-dependent climate damage
   - Redistribution via progressive taxation

### Key Insights

1. **Redistribution vs. Climate Action Tradeoff**: Resources allocated to income redistribution provide immediate utility gains (especially with high inequality aversion), while emissions abatement provides future benefits by reducing climate damage.

2. **Diminishing Marginal Utility**: Higher values of η (risk aversion) mean that redistributing income from rich to poor has greater utility benefits, favoring redistribution over abatement.

3. **Time Preference**: Higher discount rates (ρ) favor immediate redistribution over long-term climate benefits.

## Basic Usage

### Configuration Files

All model parameters are specified in JSON configuration files. See `config_baseline.json` for a complete example.

Key configuration sections:
- `scalar_parameters` - Time-invariant constants (α, δ, η, ρ, etc.)
- `time_functions` - Time-dependent functions (A(t), L(t), σ(t), etc.)
- `control_function` - Allocation policy f(t)
- `integration_parameters` - Time span and step size
- `optimization_parameters` - Optimization settings

### Output Files

Each run creates a timestamped directory:
```
./data/output/{run_name}_YYYYMMDD-HHMMSS/
├── results.csv          # Complete time series data
├── plots.pdf            # Multi-page charts
└── terminal_output.txt  # Console output
```

### Running Multiple Cases in Parallel

Launch multiple optimizations simultaneously:

```bash
python run_parallel.py "config_*.json"
```

Each optimization runs on its own CPU core, with progress logged to `terminal_output.txt` in each output directory.

## Project Structure

```
coin_equality/
├── README.md                          # This file (quick start guide)
├── README_DETAIL.md                   # Detailed technical documentation
├── CLAUDE.md                          # AI coding style guide
├── requirements.txt                   # Python dependencies
├── distribution_utilities.py          # Income distribution and utility integration
├── economic_model.py                  # Economic production and tendencies
├── parameters.py                      # Parameter definitions and configuration
├── optimization.py                    # Optimization framework
├── output.py                          # Output generation (CSV and PDF)
├── test_integration.py                # Test script for forward integration
├── run_optimization.py                # Main optimization script
├── run_parallel.py                    # Launch multiple optimizations in parallel
├── compare_results.py                 # Compare multiple optimization runs
├── comparison_utils.py                # Multi-run comparison utilities
├── visualization_utils.py             # Unified visualization functions
├── config_baseline.json               # Baseline scenario configuration
├── config_high_inequality.json        # High inequality scenario
├── data/output/                       # Output directory (timestamped subdirs)
└── coin_equality (methods) v0.1.pdf   # Detailed methods document
```

## Documentation

- **[README_DETAIL.md](README_DETAIL.md)** - Complete technical documentation including:
  - Detailed model structure with all equations
  - Complete parameter descriptions
  - Configuration file format
  - Optimization algorithms and settings
  - Implementation details
  - Performance optimizations

## References

Barrage, L., & Nordhaus, W. (2024). "Policies, projections, and the social cost of carbon: Results from the DICE-2023 model." *Proceedings of the National Academy of Sciences*, 121(13), e2312030121. https://doi.org/10.1073/pnas.2312030121

Caldeira, K., Bala, G., & Cao, L. (2023). "Climate sensitivity uncertainty and the need for energy without CO₂ emission." *Environmental Research Letters*, 18(9), 094021. https://doi.org/10.1088/1748-9326/acf949

Nordhaus, W. D. (1992). "An optimal transition path for controlling greenhouse gases." *Science*, 258(5086), 1315-1319. https://doi.org/10.1126/science.258.5086.1315

Nordhaus, W. D. (2017). "Revisiting the social cost of carbon." *Proceedings of the National Academy of Sciences*, 114(7), 1518-1523. https://doi.org/10.1073/pnas.1609244114

## License

MIT License

Copyright (c) 2025 Lamprini Papargyri, ..., and Ken Caldeira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Authors

Lamprini Papargyri, ..., and Ken Caldeira
