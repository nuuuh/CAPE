# EpiRecipe

EpiRecipe is a comprehensive 6-stage pipeline for generating diverse compartmental epidemic models. It creates structurally valid, parameterized models with realistic noise for pretraining neural forecasting systems.

## 6-Stage Pipeline

1. **Select Compartments** – Randomly choose from 12 compartments (S, I, E, R, H, V, Q, D, P, W, A, C). Always includes `S` and `I`.
2. **Determine Transition Structure** – Build valid transition paths (e.g., E→I, I→R) based on selected compartments using categorical logic.
3. **Select Mathematical Forms** – For each transition, randomly choose a valid mathematical rule from the catalog (26 rules across 5 categories).
4. **Sample Parameters** – Randomly sample epidemiological parameters (R₀, recovery rates, etc.) from configured ranges.
5. **Inject Noise** – Add realistic measurement noise at randomly sampled levels (0-20% of signal).
6. **Simulate** – Run RK4 ODE solver to generate synthetic trajectories (weekly and daily).

## Repository Contents

- **`catalog.json`** – Database of 12 compartments and 26 transition rules organized into 5 categories (infection, progression, recovery, intervention, reservoir).
- **`pipeline_new.py`** – Complete 6-stage pipeline implementation with compartment selection, transition structure building, parameter sampling, noise injection, and simulation.
- **`sample_recipes_new.py`** – Batch generation script that creates multiple diverse models with metadata.
- **`visualize.py`** – Plotting utilities for individual or batch visualization.
- **`simulations/`** – Output directory for generated CSV files and metadata.

## Transition Rule Categories

### Category 1: Infection (6 rules)
Determine how S → E/I/A with different drivers (I, A, C, H, W):
- **standard_mixing**: Weighted sum βS(I + εA + δC)/N
- **environmental_cholera**: βₘS·W/(K+W) with saturation
- **fomite_surface**: βS(I + ρW)/N combining people and surfaces
- **vector_borne_proxy**: βS(I/N)e^(-αI) with saturation
- **hospital_leakage**: βS(I + θH)/N nosocomial transmission
- **power_law_mixing**: βS^p·I^q for non-linear dynamics

### Category 2: Progression (7 rules)
Clinical evolution (E→I, I→H, I→D, etc.):
- **simple_linear**: σE constant rate
- **branching_split**: pσE→I, (1-p)σE→A
- **severity_trigger**: ηI → H
- **overflow_trigger**: ηI·sigmoid(I) panic-driven
- **chronic_failure**: χI → C for carriers
- **hospital_death**: μₕH → D
- **disease_induced_death**: μI → D

### Category 3: Recovery (4 rules)
Clearance and immunity waning:
- **constant_recovery**: γI → R
- **resource_constraints**: γₕH/(1+ωₕH) → R
- **waning_immunity**: ωR → S (SIRS dynamics)
- **carrier_clearance**: γcC → R

### Category 4: Intervention (5 rules)
Control measures:
- **campaign_mode**: ν constant vaccination S→V
- **proportional_vax**: pS proportional vaccination
- **reactive_quarantine**: τ(I)·I → Q policy-triggered
- **tracing_efficiency**: k·(S→E) contact tracing E→Q
- **prophylaxis_decay**: δP → S

### Category 5: Reservoir (3 rules)
Environmental pathogen dynamics:
- **shedding**: dW/dt += ξI
- **asymptomatic_shedding**: dW/dt += ξ(I + δA)
- **environmental_decay**: -ζW

## Quick Start

### Run Demo

```bash
cd epirecipe
conda activate cape
python pipeline_new.py
```

This demonstrates all 6 stages and saves results to:
- `simulations/demo_weekly.csv` – Weekly trajectory data
- `simulations/demo_daily.csv` – Daily trajectory data  
- `simulations/demo_plot.png` – Visualization with 4 panels

### Generate Batch of Models

```bash
# Without visualization (faster)
python sample_recipes_new.py --num-samples 100 --min-compartments 4 --max-compartments 8 --output-dir simulations/batch_100

# With visualization (generates PNG plots for each model)
python sample_recipes_new.py --num-samples 100 --min-compartments 4 --max-compartments 8 --output-dir simulations/batch_100 --visualize
```

### Programmatic Usage

```python
from epirecipe.pipeline_new import EpiRecipePipeline

# Initialize pipeline
pipeline = EpiRecipePipeline(rng_seed=2024)

# Generate a complete model (Stages 1-5)
model = pipeline.generate_model(
    num_compartments=6,
    overrides={"population": 5e5, "R0": 3.0},
    name="my_model"
)

# Inspect model structure
print(f"Compartments: {model.compartments}")
print(f"Transitions: {[str(t) for t in model.transitions]}")
print(f"Noise level: {model.noise_level:.3f}")

# Run simulation (Stage 6) with visualization
df_weekly, df_daily = pipeline.run_simulation(
    model,
    save_path=Path("output/my_model"),
    visualize=True  # Creates PNG plot
)
```

## Model Realism & Validation

The pipeline ensures realistic epidemic dynamics through several mechanisms:

**Mathematical Correctness:**
- Quarantine compartments (Q) have proper release mechanisms (default 14-day isolation)
- Hospital recovery uses resource-constrained dynamics
- All transition rates are bounded to realistic daily values
- Parameter validation ensures consistency (e.g., beta = R0 × gamma)

**Parameter Constraints:**
- R0: 1.5-3.5 (realistic for most infectious diseases)
- Infectious period (1/gamma): 3-21 days
- Latent period (1/sigma): 2-14 days
- Maximum intervention rates: 20-30% per day
- Initial conditions: <0.5% infected, <2% exposed

**Flow Balance:**
- Population conservation maintained (excluding births/deaths)
- All non-terminal compartments have outflows
- Intervention compartments (Q, V, P) have appropriate release/decay mechanisms

**R_t Dynamics:**
- Accounts for susceptible depletion (S/N)
- Reflects impact of interventions (quarantine, vaccination)
- Realistic decline over epidemic course

## Visualization

The pipeline automatically generates comprehensive 4-panel visualizations for each model when `visualize=True`:

**Panel 1: Compartment Dynamics (Weekly)** – All compartment trajectories over simulation period

**Panel 2: Infectious Compartments (Daily)** – Detailed view of disease progression (I, E, A, H, Q, C)

**Panel 3: Effective Reproduction Number (R_t)** – Time-varying R_t with epidemic threshold line

**Panel 4: Model Structure Summary** – Text panel showing:
- Selected compartments and transitions
- Key parameters (R₀, population, noise level)
- Simulation statistics (peak infections, final recovered)

### Visualization Examples

```bash
# Generate single model with visualization
python pipeline_new.py

# Batch generation with plots
python sample_recipes_new.py --num-samples 50 --visualize
```

Each simulation produces three files:
- `{name}_weekly.csv` – Weekly time series data
- `{name}_daily.csv` – Daily time series data
- `{name}_plot.png` – 4-panel visualization (if `--visualize` flag used)

# Run simulation (Stage 6)
df_weekly, df_daily = pipeline.run_simulation(
    model, 
    save_path=Path("simulations/my_model")
)
df = pipeline.simulate(selection, config, save_path="epirecipe/simulations/custom_seihrd.csv")
```

### Custom recipes

You can bypass presets by supplying explicit compartments and transitions:

```python
selection = pipeline.select_compartments(
    compartments=["S", "I", "R", "D"],
    transitions=["mass_action_infection", "recovery_linear", "disease_induced_death"],
)
selection = pipeline.apply_transition_rules(selection, additional_rules=["waning_immunity"])
config = pipeline.configure(selection, overrides={"R0": 2.7, "initial_infected": 50})
df = pipeline.simulate(selection, config)

# or let the pipeline propose a random combination
auto_selection = pipeline.generate_random_selection(num_compartments=5, num_transitions=6)
auto_config = pipeline.configure(auto_selection)
auto_df = pipeline.simulate(auto_selection, auto_config, save_path="epirecipe/simulations/random_combo.csv")
```

### Batch sampling at scale

If you want hundreds or thousands of synthetic trajectories, use the helper script:

```bash
python epirecipe/sample_recipes.py --num-samples 100 --num-compartments 6 --num-transitions 7 --seed 1337
```

Each run writes unique compartment/transition combinations under `epirecipe/simulations/random_recipes/` (configurable via `--output-dir`) and records the compartments, transitions, resolved parameters, and the aggregate `unique_recipe_count` in `metadata.json` for easy bookkeeping.

You can automatically generate plots during sampling by adding the `--plot` flag:

```bash
python epirecipe/sample_recipes.py --num-samples 100 --seed 1337 --plot --max-plots 20
```

This will create both CSV files and corresponding plots in one pass.

### Visualizing simulations

Plot individual trajectories or batch-generate figures from metadata:

```bash
# Plot a single simulation
python epirecipe/visualize.py epirecipe/simulations/seirs_demo.csv --output-dir epirecipe/simulations/plots

# Batch-plot from metadata (saves to <metadata_dir>/plots by default)
python epirecipe/visualize.py epirecipe/simulations/random_recipes/metadata.json --max-plots 10
```

Or use the Python API:

```python
from epirecipe.visualize import plot_simulation, plot_batch_simulations
from pathlib import Path

# Single plot
plot_simulation(
    Path("epirecipe/simulations/seirs_demo.csv"),
    title="SEIRS Demo",
    save_path=Path("my_plot.png"),
    show=False
)

# Batch plots
plot_batch_simulations(
    Path("epirecipe/simulations/random_recipes/metadata.json"),
    max_plots=20
)
```

### Extending the catalog

1. Append new compartments or transition objects to `catalog.json` (keep descriptions and canonical equations up to date).
2. Register the transition in `TransitionRegistry.handlers` inside `pipeline.py` so the simulator knows how to compute derivatives.
3. (Optional) Update the README or helper scripts if you want to showcase specific combinations, but explicit recipe lists are no longer required.

## Output format

Simulations return a `pandas.DataFrame` with a `day` column followed by one column per compartment in the selection. Values are non-negative floats representing counts.

---
This folder is intentionally isolated; no other project files were modified.
