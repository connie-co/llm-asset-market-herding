# Data Directory

This directory stores simulation results and generated plots. It is not tracked by git.

## Required External Data

To run simulations with real CEF data (`use_cef_data=True`), place the following files in the **parent directory** of this repository:

- `HistoricalData_BST_PRICE.csv` — BST market price history
- `HistoricalData_XBSTX_NAV.csv` — BST NAV history

These can be downloaded from [Nasdaq Historical Data](https://www.nasdaq.com/market-activity/funds-and-etfs):
- BST ticker: `BST` (market price)
- NAV ticker: `XBSTX` (net asset value)

Alternatively, set the `CEF_DATA_DIR` environment variable to point to the directory containing these files.

## Generated Subdirectories

- `results/` — Simulation output CSVs and JSON summaries
- `plots/` — Generated plot images
- `plots_v/` — Variance experiment plots
