"""Visualization functions for simulation results."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .metrics import calculate_returns

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
    ax.set_title("Distribution of Returns")
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


def main():
    """Load latest results and generate all plots."""
    import glob
    
    results_dir = Path(__file__).parent.parent / "data" / "results"
    plots_dir = Path(__file__).parent.parent / "data" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Find the latest baseline and herding files
    baseline_files = sorted(glob.glob(str(results_dir / "baseline_diverse_rounds_*.csv")))
    herding_files = sorted(glob.glob(str(results_dir / "herding_homogeneous_rounds_*.csv")))
    
    if not baseline_files or not herding_files:
        print("No results found. Run the simulation first.")
        return
    
    # Use the latest files
    baseline_path = baseline_files[-1]
    herding_path = herding_files[-1]
    
    print(f"Loading baseline: {Path(baseline_path).name}")
    print(f"Loading herding: {Path(herding_path).name}")
    
    baseline_df = pd.read_csv(baseline_path)
    herding_df = pd.read_csv(herding_path)
    
    print("\nGenerating plots...")
    
    # Generate individual plots
    plot_price_series(baseline_df, "Baseline: Price Time Series", 
                      plots_dir / "baseline_price_series.png")
    print("  - Saved baseline_price_series.png")
    
    plot_price_series(herding_df, "Herding: Price Time Series",
                      plots_dir / "herding_price_series.png")
    print("  - Saved herding_price_series.png")
    
    plot_price_comparison(baseline_df, herding_df,
                          plots_dir / "price_comparison.png")
    print("  - Saved price_comparison.png")
    
    plot_return_distribution(baseline_df, herding_df,
                             plots_dir / "return_distribution.png")
    print("  - Saved return_distribution.png")
    
    plot_herding_index(baseline_df, "Baseline: Herding Index",
                       plots_dir / "baseline_herding_index.png")
    print("  - Saved baseline_herding_index.png")
    
    plot_herding_index(herding_df, "Herding: Herding Index",
                       plots_dir / "herding_herding_index.png")
    print("  - Saved herding_herding_index.png")
    
    # Generate summary figure
    create_summary_figure(baseline_df, herding_df,
                          plots_dir / "summary.png")
    print("  - Saved summary.png")
    
    print(f"\nAll plots saved to: {plots_dir}")
    
    # Show the summary plot
    create_summary_figure(baseline_df, herding_df)
    plt.show()


if __name__ == "__main__":
    main()
