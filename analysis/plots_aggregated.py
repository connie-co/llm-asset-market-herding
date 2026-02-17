"""
Generate plots with uncertainty estimates from aggregated simulation runs.

This module creates publication-ready plots comparing baseline vs herding
conditions across different variance levels, with 95% confidence intervals.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any

from .aggregate_runs import load_config, load_all_configurations, create_summary_dataframe


# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    "baseline": "#2196F3",  # Blue
    "herding": "#F44336",   # Red
}
VARIANCE_LABELS = {5: "σ=5 (Low)", 12: "σ=12 (Med)", 20: "σ=20 (High)"}

# Output directory
ANALYSIS_DIR = Path(__file__).parent
OUTPUT_DIR = ANALYSIS_DIR.parent / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)


def plot_metric_comparison_bars(
    results: dict[str, Any],
    metric_key: str,
    title: str,
    ylabel: str,
    filename: str,
    show_values: bool = True
) -> None:
    """
    Create a grouped bar chart comparing a metric between baseline and herding.
    
    Args:
        results: Dictionary from load_all_configurations()
        metric_key: Key of the metric in aggregated_metrics
        title: Plot title
        ylabel: Y-axis label
        filename: Output filename (without extension)
        show_values: Whether to show values on bars
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    variances = [5, 12, 20]
    x = np.arange(len(variances))
    width = 0.35
    
    # Extract data for each condition
    baseline_means = []
    baseline_errors = []
    herding_means = []
    herding_errors = []
    
    for var in variances:
        # Baseline
        baseline_key = f"baseline_var{var}"
        if baseline_key in results and metric_key in results[baseline_key]["aggregated_metrics"]:
            m = results[baseline_key]["aggregated_metrics"][metric_key]
            baseline_means.append(m["mean"])
            # Error bar is distance from mean to CI bounds
            baseline_errors.append(m["ci_upper"] - m["mean"])
        else:
            baseline_means.append(0)
            baseline_errors.append(0)
        
        # Herding
        herding_key = f"herding_var{var}"
        if herding_key in results and metric_key in results[herding_key]["aggregated_metrics"]:
            m = results[herding_key]["aggregated_metrics"][metric_key]
            herding_means.append(m["mean"])
            herding_errors.append(m["ci_upper"] - m["mean"])
        else:
            herding_means.append(0)
            herding_errors.append(0)
    
    # Create bars with error bars
    bars1 = ax.bar(x - width/2, baseline_means, width, 
                   yerr=baseline_errors, capsize=5,
                   label='Baseline (Heterogeneous)', 
                   color=COLORS["baseline"], alpha=0.8)
    bars2 = ax.bar(x + width/2, herding_means, width,
                   yerr=herding_errors, capsize=5,
                   label='Herding (Homogeneous)', 
                   color=COLORS["herding"], alpha=0.8)
    
    # Labels and formatting
    ax.set_xlabel('Signal Noise Level', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANCE_LABELS[v] for v in variances])
    ax.legend(loc='upper left')
    
    # Add value labels on bars
    if show_values:
        for bars, errors in [(bars1, baseline_errors), (bars2, herding_errors)]:
            for bar, err in zip(bars, errors):
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height + err),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{filename}.png", dpi=150, bbox_inches='tight')
    print(f"  - Saved {filename}.png")
    plt.close()


def plot_volatility_comparison(results: dict[str, Any]) -> None:
    """Plot volatility comparison between conditions."""
    plot_metric_comparison_bars(
        results,
        metric_key="volatility",
        title="Price Volatility: Baseline vs Herding Condition",
        ylabel="Volatility (Std Dev of Returns, %)",
        filename="agg_volatility_comparison"
    )


def plot_herding_index_comparison(results: dict[str, Any]) -> None:
    """Plot herding index comparison between conditions."""
    plot_metric_comparison_bars(
        results,
        metric_key="mean_herding_index",
        title="Mean Herding Index: Baseline vs Herding Condition",
        ylabel="Herding Index (0-1 scale)",
        filename="agg_herding_index_comparison"
    )


def plot_flash_events_comparison(results: dict[str, Any]) -> None:
    """Plot flash events comparison between conditions."""
    plot_metric_comparison_bars(
        results,
        metric_key="total_flash_events",
        title="Flash Events (|Return| > 5%): Baseline vs Herding Condition",
        ylabel="Number of Flash Events",
        filename="agg_flash_events_comparison"
    )


def plot_price_deviation_comparison(results: dict[str, Any]) -> None:
    """Plot price deviation from true value."""
    plot_metric_comparison_bars(
        results,
        metric_key="mean_rel_deviation_pct",
        title="Price Efficiency: Mean Deviation from True Value",
        ylabel="Mean Relative Deviation (%)",
        filename="agg_price_deviation_comparison"
    )


def plot_price_series_with_bands(results: dict[str, Any], variance: int) -> None:
    """
    Plot price series over time with confidence bands across runs.
    
    Shows mean price trajectory ± 1 SE for both conditions.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for condition in ["baseline", "herding"]:
        config_key = f"{condition}_var{variance}"
        if config_key not in results:
            continue
        
        data = results[config_key]
        rounds_list = data["rounds_data"]
        
        if not rounds_list:
            continue
        
        # Find minimum length across runs (in case of different lengths)
        min_len = min(len(df) for df in rounds_list)
        
        # Stack price series
        price_matrix = np.array([df["price_after"].values[:min_len] for df in rounds_list])
        
        # Calculate mean and SE
        mean_prices = np.mean(price_matrix, axis=0)
        se_prices = np.std(price_matrix, axis=0, ddof=1) / np.sqrt(len(rounds_list))
        
        rounds = np.arange(min_len)
        
        # Plot mean line
        ax.plot(rounds, mean_prices, 
                color=COLORS[condition], 
                linewidth=2,
                label=f'{condition.capitalize()} (n={len(rounds_list)})')
        
        # Plot confidence band (±1 SE)
        ax.fill_between(rounds, 
                        mean_prices - se_prices, 
                        mean_prices + se_prices,
                        color=COLORS[condition], 
                        alpha=0.2)
    
    # Plot true value (from first baseline run)
    baseline_key = f"baseline_var{variance}"
    if baseline_key in results and results[baseline_key]["rounds_data"]:
        true_values = results[baseline_key]["rounds_data"][0]["true_value"].values[:min_len]
        ax.plot(rounds, true_values, 
                color="black", linewidth=2, linestyle="--",
                label="True Value (NAV)")
    
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Price ($)", fontsize=12)
    ax.set_title(f"Price Trajectories with Uncertainty Bands (σ={variance})", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"agg_price_series_var{variance}.png", dpi=150, bbox_inches='tight')
    print(f"  - Saved agg_price_series_var{variance}.png")
    plt.close()


def plot_all_price_series(results: dict[str, Any]) -> None:
    """Generate price series plots for all variance levels."""
    for variance in [5, 12, 20]:
        plot_price_series_with_bands(results, variance)


def plot_summary_table(results: dict[str, Any]) -> None:
    """
    Create a visual summary table as an image.
    """
    summary_df = create_summary_dataframe(results)
    
    # Select columns for the table
    display_cols = [
        "Condition", "Variance (σ)", "N Runs",
        "Volatility (%)", "Volatility (%) CI",
        "Mean Herding Index", "Mean Herding Index CI",
        "Flash Events", "Flash Events CI"
    ]
    
    # Filter to existing columns
    display_cols = [c for c in display_cols if c in summary_df.columns]
    table_df = summary_df[display_cols]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('off')
    
    # Create table
    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        cellLoc='center',
        loc='center',
        colColours=['#E3F2FD'] * len(table_df.columns)
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color rows by condition
    for i, row in enumerate(table_df.itertuples()):
        color = '#E3F2FD' if row.Condition == 'baseline' else '#FFEBEE'
        for j in range(len(display_cols)):
            table[(i + 1, j)].set_facecolor(color)
    
    plt.title("Summary Statistics with 95% Confidence Intervals", 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "agg_summary_table.png", dpi=150, bbox_inches='tight')
    print(f"  - Saved agg_summary_table.png")
    plt.close()


def plot_effect_sizes(results: dict[str, Any]) -> None:
    """
    Plot the difference (herding - baseline) for key metrics.
    Shows effect size with confidence intervals.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    metrics = [
        ("volatility", "Volatility Difference (%)", axes[0]),
        ("mean_herding_index", "Herding Index Difference", axes[1]),
        ("mean_rel_deviation_pct", "Price Deviation Difference (%)", axes[2]),
    ]
    
    variances = [5, 12, 20]
    x = np.arange(len(variances))
    
    for metric_key, ylabel, ax in metrics:
        differences = []
        combined_errors = []
        
        for var in variances:
            baseline_key = f"baseline_var{var}"
            herding_key = f"herding_var{var}"
            
            if (baseline_key in results and herding_key in results and
                metric_key in results[baseline_key]["aggregated_metrics"] and
                metric_key in results[herding_key]["aggregated_metrics"]):
                
                b = results[baseline_key]["aggregated_metrics"][metric_key]
                h = results[herding_key]["aggregated_metrics"][metric_key]
                
                diff = h["mean"] - b["mean"]
                # Combined SE for difference (assuming independence)
                combined_se = np.sqrt(b["se"]**2 + h["se"]**2)
                # 95% CI margin (approximate with z=1.96 for simplicity)
                error = 1.96 * combined_se
                
                differences.append(diff)
                combined_errors.append(error)
            else:
                differences.append(0)
                combined_errors.append(0)
        
        # Plot bars
        colors = ['green' if d > 0 else 'red' for d in differences]
        bars = ax.bar(x, differences, yerr=combined_errors, capsize=5,
                      color=colors, alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"σ={v}" for v in variances])
        ax.set_ylabel(ylabel)
        ax.set_title(f"Effect of Herding on {metric_key.replace('_', ' ').title()}")
    
    plt.suptitle("Effect Sizes: Herding - Baseline (with 95% CI)", 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "agg_effect_sizes.png", dpi=150, bbox_inches='tight')
    print(f"  - Saved agg_effect_sizes.png")
    plt.close()


def generate_all_plots(results: dict[str, Any] = None) -> None:
    """Generate all aggregated analysis plots."""
    if results is None:
        config = load_config()
        results = load_all_configurations(config)
    
    print("\nGenerating aggregated plots...")
    
    # Bar comparisons
    plot_volatility_comparison(results)
    plot_herding_index_comparison(results)
    plot_flash_events_comparison(results)
    plot_price_deviation_comparison(results)
    
    # Price series with bands
    plot_all_price_series(results)
    
    # Summary table
    plot_summary_table(results)
    
    # Effect sizes
    plot_effect_sizes(results)
    
    print(f"\nAll plots saved to: {OUTPUT_DIR}")


# Main execution
if __name__ == "__main__":
    print("Loading and aggregating simulation runs...")
    config = load_config()
    results = load_all_configurations(config)
    
    generate_all_plots(results)
    
    print("\nDone!")
