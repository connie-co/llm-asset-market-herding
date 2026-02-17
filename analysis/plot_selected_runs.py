"""
Plot selected simulation runs with explicit file control.

This module allows you to explicitly select which files to plot,
making it easier to investigate and compare specific runs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from src.data_loader import load_bst_data

# Paths
ANALYSIS_DIR = Path(__file__).parent
PROJECT_ROOT = ANALYSIS_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_DIR = PROJECT_ROOT / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# CONFIGURE YOUR RUNS HERE
# =============================================================================

# Each entry is: (label, config_name, timestamp)
# The label is what appears in the plot legend
SELECTED_RUNS = [
    # Baseline runs
    ("Baseline σ=5 (run1)", "baseline_var5", "20251228_035158"),
    ("Baseline σ=5 (run2)", "baseline_var5", "20251228_181136"),
    ("Baseline σ=12 (run1)", "baseline_var12", "20251228_072804"),
    ("Baseline σ=12 (run2)", "baseline_var12", "20251228_200251"),
    ("Baseline σ=20 (run1)", "baseline_var20", "20251228_111552"),
    ("Baseline σ=20 (run2)", "baseline_var20", "20251228_212852"),
    
    # Herding runs
    ("Herding σ=5 (run1)", "herding_var5", "20251228_053642"),
    ("Herding σ=12 (run1)", "herding_var12", "20251228_091835"),
    ("Herding σ=20 (run1)", "herding_var20", "20251228_164405"),
]

# Alternatively, compare specific pairs
COMPARISON_PAIRS = [
    # (label1, config1, ts1, label2, config2, ts2)
    ("Baseline σ=5", "baseline_var5", "20251228_035158", 
     "Herding σ=5", "herding_var5", "20251228_053642"),
    ("Baseline σ=12", "baseline_var12", "20251228_072804",
     "Herding σ=12", "herding_var12", "20251228_091835"),
    ("Baseline σ=20", "baseline_var20", "20251228_111552",
     "Herding σ=20", "herding_var20", "20251228_164405"),
]

# =============================================================================


def load_run(config_name: str, timestamp: str) -> Optional[pd.DataFrame]:
    """Load a single run's data."""
    path = RESULTS_DIR / f"{config_name}_rounds_{timestamp}.csv"
    if not path.exists():
        print(f"WARNING: File not found: {path}")
        return None
    return pd.read_csv(path)


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calculate key metrics for a run."""
    returns = df["return_pct"].values
    prices = df["price_after"].values
    true_values = df["true_value"].values
    
    # Calculate NAV returns and volatility
    nav_returns = np.diff(true_values) / true_values[:-1] * 100
    nav_volatility = np.std(nav_returns) if len(nav_returns) > 0 else 0.0

    # Calculate BST returns and volatility (Real Market Data)
    _, bst_prices = load_bst_data(paa_steps=200)
    # Trim BST data to match the length of this simulation run
    bst_prices = bst_prices[:len(df)]
    bst_returns = np.diff(bst_prices) / bst_prices[:-1] * 100
    bst_volatility = np.std(bst_returns) if len(bst_returns) > 0 else 0.0
    
    # Max drawdown
    cumulative_max = np.maximum.accumulate(prices)
    drawdowns = (prices - cumulative_max) / cumulative_max * 100
    
    return {
        "volatility": np.std(returns),
        "nav_volatility": nav_volatility,
        "bst_volatility": bst_volatility,
        "max_return": np.max(np.abs(returns)),
        "flash_events_3pct": np.sum(np.abs(returns) > 3),
        "flash_events_5pct": np.sum(np.abs(returns) > 5),
        "mean_herding_index": df["herding_index"].mean(),
        "max_drawdown": np.min(drawdowns),
        "price_deviation": np.mean(np.abs(prices - true_values) / true_values * 100),
        "net_demand_range": df["net_demand"].max() - df["net_demand"].min(),
    }


def print_run_summary(label: str, df: pd.DataFrame):
    """Print a summary of a run's metrics."""
    m = calculate_metrics(df)
    print(f"\n{label}:")
    print(f"  Price Volatility: {m['volatility']:.2f}%")
    print(f"  NAV Volatility:   {m['nav_volatility']:.2f}%")
    print(f"  BST Volatility:   {m['bst_volatility']:.2f}%")
    print(f"  Max Return: {m['max_return']:.2f}%")
    print(f"  Flash Events (>3%): {m['flash_events_3pct']}")
    print(f"  Mean Herding Index: {m['mean_herding_index']:.3f}")
    print(f"  Max Drawdown: {m['max_drawdown']:.2f}%")
    print(f"  Price Deviation: {m['price_deviation']:.2f}%")
    print(f"  Net Demand Range: {m['net_demand_range']}")


def plot_price_series(runs: list[tuple], filename: str = "selected_price_series.png"):
    """Plot price series for selected runs."""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))
    
    for i, (label, config, timestamp) in enumerate(runs):
        df = load_run(config, timestamp)
        if df is None:
            continue
        
        ax.plot(df["round"], df["price_after"], 
                label=label, color=colors[i], linewidth=1.5, alpha=0.8)
    
    # Plot true value from first run
    first_df = load_run(runs[0][1], runs[0][2])
    if first_df is not None:
        ax.plot(first_df["round"], first_df["true_value"], 
                label="True Value (NAV)", color="black", linestyle="--", linewidth=2)
    
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.set_title("Price Series: Selected Runs", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_returns_distribution(runs: list[tuple], filename: str = "selected_returns_dist.png"):
    """Plot return distributions for selected runs."""
    n_runs = len(runs)
    fig, axes = plt.subplots(1, n_runs, figsize=(4*n_runs, 4), sharey=True)
    if n_runs == 1:
        axes = [axes]
    
    for i, (label, config, timestamp) in enumerate(runs):
        df = load_run(config, timestamp)
        if df is None:
            continue
        
        returns = df["return_pct"].values
        axes[i].hist(returns, bins=30, edgecolor="black", alpha=0.7)
        axes[i].axvline(x=3, color="red", linestyle="--", label=">3% threshold")
        axes[i].axvline(x=-3, color="red", linestyle="--")
        axes[i].set_xlabel("Return (%)")
        axes[i].set_title(f"{label}\nσ={np.std(returns):.2f}%")
    
    axes[0].set_ylabel("Frequency")
    plt.suptitle("Return Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_comparison_pairs(pairs: list[tuple], filename: str = "comparison_pairs.png"):
    """Plot side-by-side comparisons of baseline vs herding."""
    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4*n_pairs))
    
    for i, (label1, config1, ts1, label2, config2, ts2) in enumerate(pairs):
        df1 = load_run(config1, ts1)
        df2 = load_run(config2, ts2)
        
        if df1 is not None:
            axes[i, 0].plot(df1["round"], df1["price_after"], label="Price", color="blue")
            axes[i, 0].plot(df1["round"], df1["true_value"], label="NAV", 
                           color="black", linestyle="--")
            axes[i, 0].set_title(f"{label1}")
            axes[i, 0].legend(loc="best")
            axes[i, 0].set_ylabel("Price ($)")
        
        if df2 is not None:
            axes[i, 1].plot(df2["round"], df2["price_after"], label="Price", color="red")
            axes[i, 1].plot(df2["round"], df2["true_value"], label="NAV",
                           color="black", linestyle="--")
            axes[i, 1].set_title(f"{label2}")
            axes[i, 1].legend(loc="best")
    
    for ax in axes[-1, :]:
        ax.set_xlabel("Round")
    
    plt.suptitle("Baseline vs Herding Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_metrics_comparison(runs: list[tuple], filename: str = "selected_metrics.png"):
    """Create a bar chart comparing metrics across runs."""
    labels = []
    volatilities = []
    nav_volatilities = []
    bst_volatilities = []
    herding_indices = []
    flash_events = []
    
    for label, config, timestamp in runs:
        df = load_run(config, timestamp)
        if df is None:
            continue
        
        m = calculate_metrics(df)
        labels.append(label)
        volatilities.append(m["volatility"])
        nav_volatilities.append(m["nav_volatility"])
        bst_volatilities.append(m["bst_volatility"])
        herding_indices.append(m["mean_herding_index"])
        flash_events.append(m["flash_events_3pct"])
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    x = np.arange(len(labels))
    width = 0.25
    
    # Volatility Comparison (Price vs NAV vs BST)
    axes[0].bar(x - width, volatilities, width, label='Price Vol', color="steelblue")
    axes[0].bar(x, nav_volatilities, width, label='NAV Vol', color="lightgray", hatch='//')
    axes[0].bar(x + width, bst_volatilities, width, label='BST Vol', color="black", alpha=0.3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel("Volatility (%)")
    axes[0].set_title("Volatility: Price vs NAV vs BST")
    axes[0].legend()
    
    # Herding Index
    axes[1].bar(x, herding_indices, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Mean Herding Index")
    axes[1].set_title("Herding Index")
    
    # Flash Events
    axes[2].bar(x, flash_events, color="seagreen")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[2].set_ylabel("Count")
    axes[2].set_title("Flash Events (>3%)")
    
    plt.suptitle("Metrics Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_price_and_returns(runs: list[tuple]):
    """Plot detailed Price and Returns analysis for each run."""
    for label, config, timestamp in runs:
        df = load_run(config, timestamp)
        if df is None:
            continue
            
        m = calculate_metrics(df)
        returns = df["return_pct"].values
        prices = df["price_after"].values
        rounds = df["round"].values
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, 
                                      gridspec_kw={'height_ratios': [2, 1]})
        
        # 1. Price Plot
        ax1.plot(rounds, prices, label='Market Price', color='blue', linewidth=2)
        ax1.plot(rounds, df['true_value'], label='True Value (NAV)', color='black', 
                 linestyle='--', alpha=0.6)
        ax1.set_title(f"Detailed Analysis: {label}", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Price ($)", fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Returns Plot
        # Returns array is typically same length as df if calculated correctly during loading,
        # but check length just in case
        if len(returns) == len(rounds):
            plot_rounds = rounds
            plot_returns = returns
        else:
            # If returns calculated manually via np.diff, it's 1 shorter
            plot_rounds = rounds[1:]
            plot_returns = returns
            
        ax2.plot(plot_rounds, plot_returns, label='Returns', color='purple', 
                 marker='o', markersize=4, alpha=0.7)
        
        # Add +/- 3 Sigma bands based on THIS run's volatility
        vol = m['volatility']
        mean_ret = np.mean(plot_returns)
        upper_3std = mean_ret + 3 * vol
        lower_3std = mean_ret - 3 * vol
        
        ax2.axhline(y=mean_ret, color='black', linestyle='-', alpha=0.3)
        ax2.axhline(y=upper_3std, color='red', linestyle='--', label=f'+3σ ({upper_3std:.2f}%)')
        ax2.axhline(y=lower_3std, color='red', linestyle='--', label=f'-3σ ({lower_3std:.2f}%)')
        
        # Fill "normal" zone
        ax2.fill_between(plot_rounds, lower_3std, upper_3std, color='green', alpha=0.05)
        
        ax2.set_ylabel("Return (%)", fontsize=12)
        ax2.set_xlabel("Round", fontsize=12)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        # Annotate Volatility
        ax2.text(0.02, 0.9, f"Price Vol = {vol:.2f}%\nNAV Vol = {m['nav_volatility']:.2f}%\nBST Vol = {m['bst_volatility']:.2f}%", 
                 transform=ax2.transAxes, fontweight='bold', 
                 bbox=dict(facecolor='white', edgecolor='purple', alpha=0.8))

        filename = f"detailed_{config}_{timestamp}.png"
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / filename, dpi=150)
        print(f"Saved: {filename}")
        plt.close()


def analyze_all_selected():
    """Analyze and summarize all selected runs."""
    print("="*60)
    print("SELECTED RUNS ANALYSIS")
    print("="*60)
    
    for label, config, timestamp in SELECTED_RUNS:
        df = load_run(config, timestamp)
        if df is not None:
            print_run_summary(label, df)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Analyzing selected runs...")
    analyze_all_selected()
    
    print("\n" + "="*60)
    print("Generating plots...")
    print("="*60)
    
    plot_price_series(SELECTED_RUNS)
    plot_returns_distribution(SELECTED_RUNS[:4])  # First 4 runs
    plot_comparison_pairs(COMPARISON_PAIRS)
    plot_metrics_comparison(SELECTED_RUNS)
    plot_price_and_returns(SELECTED_RUNS)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")
