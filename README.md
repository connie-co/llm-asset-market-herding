# LLM Asset Market Herding Simulation

**Research Question:** How does information homogeneity among AI trading agents affect price volatility and the emergence of "flash crash" dynamics?

## Project Overview

This project simulates a simple asset market where LLM-powered agents make trading decisions. We test whether information homogeneity (all agents receiving the same signal) leads to herding behavior and increased volatility, analogous to phase transitions in the Ising model.

### Physics Analogy (Ising Model)

| Economic Concept | Physics Analog |
|------------------|----------------|
| Agent | Spin |
| Buy/Sell decision | Spin up/down |
| Information signal | External field |
| Herding tendency | Ferromagnetic coupling (J > 0) |
| Flash crash | Phase transition |


# Using CEF NAV Data in Simulations

## Overview

The simulation now supports using real-world Closed-End Fund (CEF) Net Asset Value (NAV) data as the "true value" series instead of a random walk. This allows you to test how agents behave when tracking a real financial asset.

## Quick Start

### Option 1: Use CEF Data (BST NAV)

To run simulations with BST CEF NAV data, set `use_cef_data=True` in your configuration:

```python
from src.config import SimulationConfig

# Create config with CEF data
sim_config = SimulationConfig(
    use_cef_data=True,
    n_rounds=300,  # Must match the PAA downsampled length
    n_agents=10,
    signal_noise_std=5.0,
)
```

Or set it via environment variable in your `.env` file:
```
USE_CEF_DATA=true
N_ROUNDS=300
```

### Option 2: Use Random Walk (Default)

To use the traditional random walk model:

```python
sim_config = SimulationConfig(
    use_cef_data=False,  # This is the default
    n_rounds=35,
    initial_true_value=100.0,
    true_value_drift=0.0,
    true_value_volatility=2.0,
)
```

## How It Works

1. **CEF Data Mode** (`use_cef_data=True`):
   - Loads BST price and NAV data from CSV files
   - Merges on matching dates
   - Applies Piecewise Aggregate Approximation (PAA) to downsample to 300 steps
   - Uses the NAV as the "true value" that agents try to track
   - The market price starts at the initial price and evolves based on agent decisions

2. **Random Walk Mode** (`use_cef_data=False`):
   - Generates a stochastic random walk with specified drift and volatility
   - Traditional simulation approach

## Data Requirements

The CEF data files must be located in the project root:
- `HistoricalData_BST_PRICE.csv` - BST market price history
- `HistoricalData_XBSTX_NAV.csv` - BST NAV history

These can be downloaded from Yahoo Finance:
- BST ticker: `BST` (market price)
- NAV ticker: `XBSTX` (net asset value)

## Configuration Parameters

### When using CEF data (`use_cef_data=True`):
- `n_rounds`: Should be set to 300 (the PAA downsampled length)
- `initial_price`: Starting market price (agents will trade around this)
- `signal_noise_std`: Controls information quality for agents
- `true_value_drift` and `true_value_volatility`: **Ignored** (real data is used)

### When using random walk (`use_cef_data=False`):
- `n_rounds`: Any value (typically 35-50 for quick tests)
- `initial_true_value`: Starting true value
- `true_value_drift`: Mean change per round
- `true_value_volatility`: Standard deviation of changes
- `signal_noise_std`: Controls information quality for agents

## Example: Running Variance Experiments with CEF Data

```python
from src.config import SimulationConfig, ExperimentConfig, VARIANCE_LIST
from src.simulation import run_experiment
from src.data_loader import get_true_value_series

# Load CEF data once for all experiments
sim_config = SimulationConfig(use_cef_data=True, n_rounds=300)
true_value_series = get_true_value_series(
    n_steps=300,
    use_cef_data=True,
)

# Run experiments with different noise levels
for variance in VARIANCE_LIST:
    modified_config = sim_config.model_copy()
    modified_config.signal_noise_std = variance
    
    # Baseline experiment
    baseline_exp = ExperimentConfig(
        experiment_name=f"cef_baseline_var{int(variance)}",
        homogeneous_information=False,
    )
    run_experiment(modified_config, baseline_exp, true_value_series=true_value_series)
    
    # Herding experiment
    herding_exp = ExperimentConfig(
        experiment_name=f"cef_herding_var{int(variance)}",
        homogeneous_information=True,
    )
    run_experiment(modified_config, herding_exp, true_value_series=true_value_series)
```

## Switching Between Modes

You can easily switch between CEF data and random walk by changing a single parameter:

```python
# Test with CEF data
sim_config.use_cef_data = True
sim_config.n_rounds = 300

# Test with random walk
sim_config.use_cef_data = False
sim_config.n_rounds = 35
```

## Notes

- **CEF data provides realistic volatility**: The BST NAV shows real market dynamics including trends, reversals, and varying volatility
- **300 steps is optimal**: This balances detail (from ~1256 trading days) with computational efficiency
- **Fair comparisons**: All experiments use the same true value series (whether CEF or random walk)
- **Discount/Premium analysis**: You can compare market price vs. NAV to see if agents create discounts or premiums similar to real CEFs

## Project Structure

```
llm-asset-market-herding/
├── README.md                 # This file
├── DEVLOG.md                 # Step-by-step development log
├── pyproject.toml            # Project dependencies (uv/pip)
├── .env.example              # Template for API keys
├── .gitignore                # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── agent.py              # LLM agent class
│   ├── market.py             # Market environment
│   ├── simulation.py         # Main simulation loop
│   └── config.py             # Configuration parameters
├── experiments/
│   ├── __init__.py
│   ├── exp1_baseline.py      # Diverse information scenario
│   └── exp2_homogeneous.py   # Homogeneous information scenario
├── analysis/
│   ├── __init__.py
│   ├── metrics.py            # Volatility, returns analysis
│   └── plots.py              # Visualization functions
├── data/
│   └── results/              # Simulation outputs (CSV, JSON)
├── notebooks/
│   └── analysis.ipynb        # Interactive analysis
└── tests/
    └── test_market.py        # Unit tests
```

## Setup

### Prerequisites

- Python 3.11+
- Google Gemini API key (free tier available)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd llm-asset-market-herding

# Create virtual environment and install dependencies
uv venv
uv pip install -e .

# Or with pip
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -e .
```

### Configuration

1. Copy `.env.example` to `.env`
2. Add your Gemini API key

```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

## Running Experiments

```bash
# Run baseline experiment (diverse information)
python -m experiments.exp1_baseline

# Run homogeneous information experiment
python -m experiments.exp2_homogeneous

# Run full comparison
python -m simulation.run_all
```

## Key Metrics

1. **Price Volatility:** Standard deviation of returns
2. **Flash Crash Frequency:** Number of price drops > 5% in a single period
3. **Herding Index:** Correlation of agent decisions
4. **Price Deviation:** Distance from fundamental value

## References

- Rio-Chanona et al. (2025) "Can Generative AI agents behave like humans?" arXiv:2505.07457
- Chen et al. (2015) "Agent-based model with multi-level herding" Nature Scientific Reports
- Gao et al. (2024) "High-Frequency Financial Market Simulation and Flash Crash Scenarios"

## License

MIT License - See LICENSE file

## Author

C Cendon 
