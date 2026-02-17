"""Visualization functions for simulation results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import calculate_returns
from src.data_loader import load_bst_data

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100


def plot_price_series(
    df: pd.DataFrame,
    title: str = "Price Time Series",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot price and true value over time.

    Args:
        df: DataFrame with 'round', 'price_after', 'true_value' columns
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["round"], df["price_after"], label="Market Price", linewidth=2)
    ax.plot(
        df["round"],
        df["true_value"],
        label="True Value",
        linestyle="--",
        alpha=0.7,
        linewidth=2,
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Price ($)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_price_comparison(
    baseline_df: pd.DataFrame,
    herding_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare price series between baseline and herding experiments.

    Args:
        baseline_df: Results from baseline experiment
        herding_df: Results from herding experiment
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Baseline
    axes[0].plot(
        baseline_df["round"],
        baseline_df["price_after"],
        label="Market Price",
        linewidth=2,
    )
    axes[0].plot(
        baseline_df["round"],
        baseline_df["true_value"],
        label="True Value",
        linestyle="--",
        alpha=0.7,
    )
    axes[0].set_ylabel("Price ($)")
    axes[0].set_title("Baseline: Diverse Information")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Herding
    axes[1].plot(
        herding_df["round"],
        herding_df["price_after"],
        label="Market Price",
        linewidth=2,
        color="red",
    )
    axes[1].plot(
        herding_df["round"],
        herding_df["true_value"],
        label="True Value",
        linestyle="--",
        alpha=0.7,
    )
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Price ($)")
    axes[1].set_title("Herding: Homogeneous Information")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_volatility_comparison(
    baseline_volatility: float,
    herding_volatility: float,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Bar chart comparing volatility between experiments.

    Args:
        baseline_volatility: Volatility from baseline experiment
        herding_volatility: Volatility from herding experiment
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    experiments = ["Baseline\n(Diverse Info)", "Herding\n(Homogeneous Info)"]
    volatilities = [baseline_volatility, herding_volatility]
    colors = ["steelblue", "indianred"]

    bars = ax.bar(experiments, volatilities, color=colors, edgecolor="black")

    # Add value labels on bars
    for bar, vol in zip(bars, volatilities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{vol:.2f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Volatility (Std Dev of Returns, %)")
    ax.set_title("Price Volatility Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_return_distribution(
    baseline_df: pd.DataFrame,
    herding_df: pd.DataFrame,
    variance: Optional[float] = None,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare return distributions between experiments.

    Args:
        baseline_df: Results from baseline experiment
        herding_df: Results from herding experiment
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_returns = calculate_returns(baseline_df["price_after"].values)
    herding_returns = calculate_returns(herding_df["price_after"].values)

    ax.hist(
        baseline_returns,
        bins=20,
        alpha=0.6,
        label="Baseline",
        color="steelblue",
        edgecolor="black",
    )
    ax.hist(
        herding_returns,
        bins=20,
        alpha=0.6,
        label="Herding",
        color="indianred",
        edgecolor="black",
    )

    ax.axvline(x=-5, color="red", linestyle="--", label="Flash Crash Threshold (-5%)")
    ax.axvline(x=5, color="green", linestyle="--", label="Flash Rally Threshold (+5%)")

    ax.set_xlabel("Return (%)")
    ax.set_ylabel("Frequency")
    title = "Distribution of Returns"
    if variance is not None: #way of adding text with conditions
        title += f" (Noise Std Dev = {variance})"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_herding_index(
    df: pd.DataFrame,
    title: str = "Herding Index Over Time",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot herding index over time.

    Args:
        df: DataFrame with 'round' and 'herding_index' columns
        title: Plot title
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df["round"], df["herding_index"], linewidth=2, color="purple")
    ax.fill_between(
        df["round"], 0, df["herding_index"], alpha=0.3, color="purple"
    )

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="50% Threshold")
    ax.axhline(y=1.0, color="darkred", linestyle="--", alpha=0.7, label="Complete Herding")

    ax.set_xlabel("Round")
    ax.set_ylabel("Herding Index")
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_agent_actions_heatmap(
    decisions_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a heatmap of agent actions over time.

    Args:
        decisions_df: DataFrame with 'round', 'agent_id', 'action' columns
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    # Pivot to get agents x rounds matrix
    action_map = {"BUY": 1, "HOLD": 0, "SELL": -1}
    decisions_df["action_numeric"] = decisions_df["action"].map(action_map)

    pivot = decisions_df.pivot(
        index="agent_id", columns="round", values="action_numeric"
    )

    fig, ax = plt.subplots(figsize=(14, 8))

    sns.heatmap(
        pivot,
        cmap="RdYlGn",
        center=0,
        cbar_kws={"label": "Action (Sell=-1, Hold=0, Buy=1)"},
        ax=ax,
    )

    ax.set_xlabel("Round")
    ax.set_ylabel("Agent")
    ax.set_title("Agent Actions Over Time")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def create_summary_figure(
    baseline_df: pd.DataFrame,
    herding_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Create a comprehensive summary figure with multiple subplots.

    Args:
        baseline_df: Results from baseline experiment
        herding_df: Results from herding experiment
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12))

    # Price comparison (top row, full width)
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(baseline_df["round"], baseline_df["price_after"], label="Baseline", linewidth=2)
    ax1.plot(herding_df["round"], herding_df["price_after"], label="Herding", linewidth=2)
    ax1.plot(
        baseline_df["round"],
        baseline_df["true_value"],
        label="True Value",
        linestyle="--",
        alpha=0.5,
    )
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Price ($)")
    ax1.set_title("Price Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Volatility comparison (top right)
    ax2 = fig.add_subplot(2, 2, 2)
    baseline_returns = calculate_returns(baseline_df["price_after"].values)
    herding_returns = calculate_returns(herding_df["price_after"].values)
    ax2.bar(
        ["Baseline", "Herding"],
        [np.std(baseline_returns), np.std(herding_returns)],
        color=["steelblue", "indianred"],
        edgecolor="black",
    )
    ax2.set_ylabel("Volatility (%)")
    ax2.set_title("Volatility Comparison")
    ax2.grid(True, alpha=0.3, axis="y")

    # Return distribution (bottom left)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(baseline_returns, bins=15, alpha=0.6, label="Baseline", color="steelblue")
    ax3.hist(herding_returns, bins=15, alpha=0.6, label="Herding", color="indianred")
    ax3.axvline(x=-5, color="red", linestyle="--", alpha=0.7)
    ax3.axvline(x=5, color="green", linestyle="--", alpha=0.7)
    ax3.set_xlabel("Return (%)")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Return Distribution")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Herding index comparison (bottom right)
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.plot(baseline_df["round"], baseline_df["herding_index"], label="Baseline", linewidth=2)
    ax4.plot(herding_df["round"], herding_df["herding_index"], label="Herding", linewidth=2)
    ax4.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax4.set_xlabel("Round")
    ax4.set_ylabel("Herding Index")
    ax4.set_title("Herding Index Over Time")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_price_series_by_variance(
    data_by_variance: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Compare price series across different variance levels for baseline and herding.

    Args:
        data_by_variance: Dict with data, e.g., {5: {'baseline': df, 'herding': df}}
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    variances = sorted(data_by_variance.keys())
    n_variances = len(variances)

    # Use a color palette
    palette = sns.color_palette("viridis", n_variances)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True, sharey=True)

    # --- Plot True Value (it's the same for all) ---
    # Get the first available dataframe to extract the true value series
    first_variance = variances[0]
    first_df = data_by_variance[first_variance]['baseline']
    axes[0].plot(
        first_df["round"],
        first_df["true_value"],
        label="True Value (NAV)",
        color="black",
        linestyle="--",
        alpha=0.7,
    )
    axes[1].plot(
        first_df["round"],
        first_df["true_value"],
        label="True Value (NAV)",
        color="black",
        linestyle="--",
        alpha=0.7,
    )

    # --- Plot BST Market Price for comparison ---
    nav_paa, price_paa = load_bst_data(paa_steps=200)
    n_rounds = len(first_df)
    price_trimmed = price_paa[:n_rounds]
    
    axes[0].plot(
        first_df["round"],
        price_trimmed,
        label="BST Market Price",
        color="red",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
    )
    axes[1].plot(
        first_df["round"],
        price_trimmed,
        label="BST Market Price",
        color="red",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
    )

    # --- Plot Baseline and Herding series for each variance ---
    for i, variance in enumerate(variances):
        color = palette[i]
        label = f"Noise Std Dev = {variance}"

        # Baseline plot
        if 'baseline' in data_by_variance[variance]:
            df = data_by_variance[variance]['baseline']
            axes[0].plot(df["round"], df["price_after"], label=label, color=color, linewidth=1.5)

        # Herding plot
        if 'herding' in data_by_variance[variance]:
            df = data_by_variance[variance]['herding']
            axes[1].plot(df["round"], df["price_after"], label=label, color=color, linewidth=1.5)

    axes[0].set_title("Baseline Condition (Diverse Information)")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    axes[1].set_title("Herding Condition (Homogeneous Information)")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Price ($)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    fig.suptitle("Price Series Comparison by Information Noise Level", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_volatility_by_variance(
    data_by_variance: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot how volatility changes as a function of information noise.

    Args:
        data_by_variance: Dict with data, e.g., {5: {'baseline': df, 'herding': df}}
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    variances = sorted(data_by_variance.keys())
    baseline_vols = []
    herding_vols = []
    nav_vols = []
    bst_vols=[]
    
    # Calculate BST Volatility (Real Market Data)
    # We calculate this once and reuse it for all variance levels since it's the benchmark
    
    # 1. Get a sample dataframe to determine n_rounds
    if not variances:
        return  # No data
        
    first_var = variances[0]
    if 'baseline' in data_by_variance[first_var]:
        sample_df = data_by_variance[first_var]['baseline']
    elif 'herding' in data_by_variance[first_var]:
        sample_df = data_by_variance[first_var]['herding']
    else:
        sample_df = None
        
    # 2. Calculate BST volatility if we have a sample
    bst_vol_value = np.nan
    if sample_df is not None:
        n_rounds = len(sample_df)
        
        # Load real BST data (returns tuple: nav, price)
        # We only need the price (index 1)
        _, bst_prices = load_bst_data(paa_steps=200)
        
        # TRIM it to match the simulation length!
        bst_prices_trimmed = bst_prices[:n_rounds]
        
        # Calculate returns and volatility
        bst_returns = calculate_returns(bst_prices_trimmed)
        bst_vol_value = np.std(bst_returns)
    
    # 3. Create a list of the same value for plotting (constant line)
    bst_vols = [bst_vol_value] * len(variances)
    
    for var in variances:
        # Calculate baseline volatility
        if 'baseline' in data_by_variance[var]:
            baseline_df = data_by_variance[var]['baseline']
            baseline_returns = calculate_returns(baseline_df["price_after"].values)
            baseline_vols.append(np.std(baseline_returns))
            
            # Calculate NAV volatility (from baseline run)
            if "true_value" in baseline_df.columns:
                nav_returns = calculate_returns(baseline_df["true_value"].values)
                nav_vols.append(np.std(nav_returns))
            else:
                nav_vols.append(np.nan)
        else:
            baseline_vols.append(np.nan)
            # Try to get NAV volatility from herding run if baseline is missing
            if 'herding' in data_by_variance[var] and "true_value" in data_by_variance[var]['herding'].columns:
                herding_df = data_by_variance[var]['herding']
                nav_returns = calculate_returns(herding_df["true_value"].values)
                nav_vols.append(np.std(nav_returns))
                
            else:
                nav_vols.append(np.nan)

        # Calculate herding volatility
        if 'herding' in data_by_variance[var]:
            herding_df = data_by_variance[var]['herding']
            herding_returns = calculate_returns(herding_df["price_after"].values)
            herding_vol = np.std(herding_returns)
            herding_vols.append(herding_vol)
            print(f'\n=== HERDING VAR {var} ===\nReturns: {herding_returns}\nVolatility: {herding_vol:.4f}%')
        else:
            herding_vols.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(variances, baseline_vols, marker='o', linestyle='-', label="Baseline (Diverse Info)")
    ax.plot(variances, herding_vols, marker='s', linestyle='-', label="Herding (Homogeneous Info)")
    
    # Plot NAV volatility if available
    if not all(np.isnan(nav_vols)):
        # Plot as a dashed line (should be roughly constant if same NAV data used)
        ax.plot(variances, nav_vols, marker='^', linestyle='--', color='gray', alpha=0.7, label="True Value (NAV) Volatility")
    # Plot BST Volatility
    if not all(np.isnan(bst_vols)):
        ax.plot(variances, bst_vols, marker='*', linestyle=':', color='black', 
                alpha=0.8, label="Real Market (BST) Volatility")
    ax.set_xlabel("Signal Noise Standard Deviation")
    ax.set_ylabel("Price Volatility (Std Dev of Returns, %)")
    ax.set_title("Market Volatility vs. Information Noise")
    ax.set_xticks(variances) # Ensure ticks are exactly at our tested variance levels
    ax.legend()
    ax.grid(True, alpha=0.4)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_herding_by_variance(
    data_by_variance: dict,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot how the average herding index changes as a function of information noise.

    Args:
        data_by_variance: Dict with data, e.g., {5: {'baseline': df, 'herding': df}}
        save_path: Path to save figure (optional)

    Returns:
        Matplotlib figure
    """
    variances = sorted(data_by_variance.keys())
    baseline_herding = []
    herding_herding = []

    for var in variances:
        # Calculate baseline average herding
        if 'baseline' in data_by_variance[var]:
            baseline_df = data_by_variance[var]['baseline']
            baseline_herding.append(baseline_df["herding_index"].mean())
        else:
            baseline_herding.append(np.nan)

        # Calculate herding average herding
        if 'herding' in data_by_variance[var]:
            herding_df = data_by_variance[var]['herding']
            herding_herding.append(herding_df["herding_index"].mean())
        else:
            herding_herding.append(np.nan)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(variances, baseline_herding, marker='o', linestyle='-', label="Baseline (Diverse Info)")
    ax.plot(variances, herding_herding, marker='s', linestyle='-', label="Herding (Homogeneous Info)")

    ax.set_xlabel("Signal Noise Standard Deviation")
    ax.set_ylabel("Average Herding Index")
    ax.set_title("Average Herding vs. Information Noise")
    ax.set_xticks(variances)
    ax.set_ylim(0, 1) # Herding index is between 0 and 1
    ax.legend()
    ax.grid(True, alpha=0.4)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def main():
    """Load all variance experiment results and generate comparison plots."""
    import glob
    import re

    results_dir = Path(__file__).parent.parent / "data" / "results"
    plots_dir = Path(__file__).parent.parent / "data" / "plots_v"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Find all csv files from the variance experiments
    all_files = glob.glob(str(results_dir / "*_var*.csv"))

    if not all_files:
        print("No variance experiment results found. Run the simulation first.")
        return

    # Dictionary to hold all dataframes, structured by variance
    # e.g., {5: {'baseline': df, 'herding': df}, 20: {...}}
    data_by_variance = {}

    # Regex to parse filenames like 'baseline_var5_rounds_...csv'
    file_pattern = re.compile(r"(baseline|herding)_var(\d+)_.*\.csv")

    for file_path in all_files:
        filename = Path(file_path).name
        match = file_pattern.match(filename)

        if match:
            condition = match.group(1)
            variance = int(match.group(2))

            print(f"Loading {condition} data for variance {variance} from {filename}")

            df = pd.read_csv(file_path)

            if variance not in data_by_variance:
                data_by_variance[variance] = {}

            data_by_variance[variance][condition] = df

    print("\nData loaded for variances:", sorted(data_by_variance.keys()))

    # --- Generate and save plots ---
    print("\nGenerating plots...")

    # Plot 1: Price Series Comparison
    price_plot_path = plots_dir / "variance_price_series_comparison.png"
    plot_price_series_by_variance(data_by_variance, save_path=price_plot_path)
    print(f"  - Saved {price_plot_path.name}")

    # Plot 2: Volatility Comparison
    vol_plot_path = plots_dir / "variance_volatility_comparison.png"
    plot_volatility_by_variance(data_by_variance, save_path=vol_plot_path)
    print(f"  - Saved {vol_plot_path.name}")

    # Plot 3: Herding Index Comparison
    herding_plot_path = plots_dir / "variance_herding_comparison.png"
    plot_herding_by_variance(data_by_variance, save_path=herding_plot_path)
    print(f"  - Saved {herding_plot_path.name}")

    # Plot 4: Return Distribution for each variance level
    for var in sorted(data_by_variance.keys()):
        if 'baseline' in data_by_variance[var] and 'herding' in data_by_variance[var]:
            return_plot_path = plots_dir / f"variance_{var}_return_distribution.png"
            plot_return_distribution(
                data_by_variance[var]['baseline'],
                data_by_variance[var]['herding'],
                variance=var,
                save_path=return_plot_path
            )
            print(f"  - Saved {return_plot_path.name}")

    print("\nAll plots generated successfully!")
    
    # Display all plots
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
