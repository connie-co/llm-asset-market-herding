# Development Log

This document tracks the step-by-step development of the LLM Asset Market Herding simulation.

---

## Phase 1: Project Setup

### Step 1.1: Initialize Project Structure ✅
**Date:** 2025-12-04

**What we did:**
- Created project directory with clean structure
- Set up `pyproject.toml` for dependency management
- Created README with project overview
- Set up `.gitignore` and `.env.example`

**Files created:**
- `README.md`
- `DEVLOG.md` (this file)
- `pyproject.toml`
- `.gitignore`
- `.env.example`

---

### Step 1.2: Core Dependencies
**Status:** ✅ COMPLETE

**Dependencies defined in `pyproject.toml`:**
- `google-generativeai>=0.8.0` - Gemini API client
- `pandas>=2.2.0` - Data handling
- `numpy>=1.26.0` - Numerical operations
- `matplotlib>=3.8.0` - Plotting
- `seaborn>=0.13.0` - Statistical visualization
- `python-dotenv>=1.0.0` - Environment variables
- `pydantic>=2.5.0` - Configuration validation
- `pydantic-settings>=2.1.0` - Settings management
- `tqdm>=4.66.0` - Progress bars
- `pytest>=7.4.0` (dev) - Testing

**To install:**
```bash
pip install -e .
# or with dev dependencies:
pip install -e ".[dev]"
```

---

## Phase 2: Core Implementation

### Step 2.1: Configuration Module
**Status:** ✅ COMPLETE

**File:** `src/config.py`

**Implemented:**
- `SimulationConfig` - Main simulation parameters using Pydantic Settings
- `ExperimentConfig` - Experiment-specific parameters
- Pre-defined configs: `BASELINE_EXPERIMENT`, `HERDING_EXPERIMENT`
- Automatic `.env` file loading for API keys

**Key parameters:**
- `n_agents`: Number of trading agents (default: 10)
- `n_rounds`: Number of trading rounds (default: 50)
- `initial_price`: Starting asset price (default: 100)
- `true_value_volatility`: Std dev of true value changes
- `signal_noise_std`: Std dev of information noise
- `price_impact`: How much each trade affects price
- `homogeneous_information`: Key experimental variable

---

### Step 2.2: Market Environment
**Status:** ✅ COMPLETE

**File:** `src/market.py`

**Implemented:**
- `Action` enum: BUY, SELL, HOLD with numeric conversion
- `MarketState` dataclass: Current market snapshot
- `RoundResult` dataclass: Results from each trading round
- `Market` class with methods:
  - `reset()`: Initialize/reset market state
  - `evolve_true_value()`: Random walk with drift
  - `generate_signals()`: Diverse or homogeneous signals
  - `process_actions()`: Update price based on agent actions
  - `get_results_summary()`: Compute volatility, flash crashes, etc.

**Price update rule:**
```python
price_change = price_impact * net_demand / n_agents
price_change = clip(price_change, -max_change, +max_change)
new_price = max(1.0, old_price + price_change)
```

---

### Step 2.3: LLM Agent
**Status:** ✅ COMPLETE

**File:** `src/agent.py`

**Implemented:**
- `AgentDecision` dataclass: Action + reasoning + metadata
- `TradingAgent` class:
  - Configurable Gemini model and temperature
  - System prompt for trading context
  - `_build_prompt()`: Constructs prompt with price, signal, history
  - `_parse_response()`: Extracts JSON action from LLM response
  - `decide()`: Main decision method
- Helper functions:
  - `configure_gemini()`: Set up API key
  - `create_agents()`: Factory for multiple agents

**Agent prompt includes:**
- Current market price
- Private signal about true value
- Recent price history (configurable)
- Price trend indicator

---

### Step 2.4: Simulation Loop
**Status:** ✅ COMPLETE

**File:** `src/simulation.py`

**Implemented:**
- `Simulation` class orchestrating everything:
  - `run_round()`: Execute single trading round
  - `run()`: Full simulation with progress bar
  - `save_results()`: Export to CSV/JSON
  - `get_results_dataframe()`: Pandas DataFrame output
- `run_experiment()`: Convenience function
- `main()`: Entry point running both experiments

**Each round:**
1. Evolve true value (random walk)
2. Generate signals (diverse or homogeneous)
3. Get market state
4. Query each agent via Gemini API
5. Process actions, update price
6. Log round data and agent decisions

---

## Phase 3: Experiments

### Step 3.1: Baseline Experiment (Diverse Information)
**Status:** PENDING

**Setup:**
- Each agent receives independent noisy signal
- Signal_i = TrueValue + noise_i, where noise_i ~ N(0, σ)

**Expected outcome:**
- Moderate volatility
- Price tracks true value with noise
- No extreme crashes

---

### Step 3.2: Homogeneous Information Experiment
**Status:** PENDING

**Setup:**
- All agents receive the SAME signal
- Signal = TrueValue + noise (same noise for all)

**Expected outcome:**
- Higher volatility
- Herding behavior (all buy or all sell together)
- Potential "flash crash" dynamics

---

## Phase 4: Analysis

### Step 4.1: Metrics Implementation
**Status:** PENDING

**Metrics to compute:**
1. **Volatility:** `std(returns)` where `return_t = (price_t - price_{t-1}) / price_{t-1}`
2. **Flash crash count:** Number of periods where `return < -5%`
3. **Herding index:** `|mean(actions)|` where BUY=+1, SELL=-1, HOLD=0
4. **Price efficiency:** `mean(|price - true_value|)`

---

### Step 4.2: Visualization
**Status:** PENDING

**Plots to create:**
1. Price time series (with true value overlay)
2. Volatility comparison (bar chart)
3. Agent decision heatmap
4. Return distribution (histogram)

---

## Phase 5: Documentation & Polish

### Step 5.1: Results Summary
**Status:** PENDING

### Step 5.2: Code Cleanup
**Status:** PENDING

### Step 5.3: Final README
**Status:** PENDING

---

## Notes & Decisions

### Design Decisions

1. **Why Gemini?**
   - Free tier available
   - Simple API
   - Good for demonstration purposes

2. **Why single asset?**
   - Simplest possible market
   - Clear signal-to-noise in results
   - Easy to visualize

3. **Why 10-20 agents?**
   - Enough for statistical patterns
   - Manageable API costs
   - Fast simulation time

### Potential Extensions

- Add agent memory (past trades influence future)
- Add agent personas (risk-averse, aggressive)
- Add market maker agent
- Compare different LLMs (Gemini vs GPT vs Claude)

---

## Issues & Solutions

*This section will be updated as we encounter and solve problems.*

---

## References Used

1. Rio-Chanona et al. (2025) - LLM agent behavior in markets
2. Zhao et al. (2015) - Herding in ABMs
3. Ising model literature - Phase transition analogy
