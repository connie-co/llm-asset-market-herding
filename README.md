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
