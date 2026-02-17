"""
Aggregate multiple simulation runs for statistical analysis.

This module loads simulation results based on a config file,
calculates per-run metrics, and aggregates them with uncertainty estimates.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any
from scipy import stats


# Path setup
ANALYSIS_DIR = Path(__file__).parent
PROJECT_ROOT = ANALYSIS_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
CONFIG_FILE = ANALYSIS_DIR / "runs_config.json"


def load_config(config_path: Path = CONFIG_FILE) -> dict:
    """Load the runs configuration file."""
    with open(config_path, "r") as f:
        return json.load(f)


def get_file_path(config_name: str, timestamp: str, file_type: str = "rounds") -> Path:
    """
    Construct the file path for a specific run.
    
    Args:
        config_name: e.g., "baseline_var5"
        timestamp: e.g., "20251228_181136"
        file_type: "rounds" or "decisions"
    
    Returns:
        Path to the CSV file
    """
    filename = f"{config_name}_{file_type}_{timestamp}.csv"
    return RESULTS_DIR / filename


def load_single_run(config_name: str, timestamp: str) -> dict[str, pd.DataFrame]:
    """
    Load both rounds and decisions data for a single run.
    
    Returns:
        Dictionary with 'rounds' and 'decisions' DataFrames
    """
    rounds_path = get_file_path(config_name, timestamp, "rounds")
    decisions_path = get_file_path(config_name, timestamp, "decisions")
    
    data = {}
    
    if rounds_path.exists():
        data["rounds"] = pd.read_csv(rounds_path)
    else:
        print(f"Warning: Rounds file not found: {rounds_path}")
        data["rounds"] = None
        
    if decisions_path.exists():
        data["decisions"] = pd.read_csv(decisions_path)
    else:
        print(f"Warning: Decisions file not found: {decisions_path}")
        data["decisions"] = None
    
    return data


def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate percentage returns from price series."""
    if len(prices) < 2:
        return np.array([])
    return np.diff(prices) / prices[:-1] * 100


def calculate_run_metrics(rounds_df: pd.DataFrame) -> dict[str, float]:
    """
    Calculate all metrics for a single simulation run.
    
    Args:
        rounds_df: DataFrame with columns like 'price_after', 'true_value', 
                   'herding_index', 'n_buys', 'n_sells', etc.
    
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}
    
    # Price series
    prices = rounds_df["price_after"].values
    true_values = rounds_df["true_value"].values
    
    # Returns
    returns = calculate_returns(prices)
    
    # Volatility (std of returns)
    metrics["volatility"] = float(np.std(returns)) if len(returns) > 0 else 0.0
    
    # Mean and max return
    metrics["mean_return"] = float(np.mean(returns)) if len(returns) > 0 else 0.0
    metrics["max_return"] = float(np.max(np.abs(returns))) if len(returns) > 0 else 0.0
    
    # Flash events (returns exceeding threshold)
    # Using 3% threshold - more appropriate for this simulation's price dynamics
    threshold = 3.0
    metrics["flash_crashes"] = int(np.sum(returns < -threshold))
    metrics["flash_rallies"] = int(np.sum(returns > threshold))
    metrics["total_flash_events"] = metrics["flash_crashes"] + metrics["flash_rallies"]
    
    # Herding index
    if "herding_index" in rounds_df.columns:
        metrics["mean_herding_index"] = float(rounds_df["herding_index"].mean())
        metrics["max_herding_index"] = float(rounds_df["herding_index"].max())
    else:
        # Calculate from n_buys, n_sells, n_holds
        n_agents = rounds_df["n_buys"] + rounds_df["n_sells"] + rounds_df["n_holds"]
        net_demand = rounds_df["n_buys"] - rounds_df["n_sells"]
        herding_index = np.abs(net_demand) / n_agents
        metrics["mean_herding_index"] = float(herding_index.mean())
        metrics["max_herding_index"] = float(herding_index.max())
    
    # Price efficiency (deviation from true value)
    deviations = np.abs(prices - true_values)
    relative_deviations = deviations / true_values * 100
    metrics["mean_abs_deviation"] = float(np.mean(deviations))
    metrics["mean_rel_deviation_pct"] = float(np.mean(relative_deviations))
    metrics["rmse"] = float(np.sqrt(np.mean((prices - true_values) ** 2)))
    
    # Final price vs final true value
    metrics["final_price"] = float(prices[-1])
    metrics["final_true_value"] = float(true_values[-1])
    metrics["final_deviation_pct"] = float((prices[-1] - true_values[-1]) / true_values[-1] * 100)
    
    # Max drawdown (largest cumulative drop from a peak)
    # This measures the worst "crash" in terms of cumulative price decline
    cumulative_max = np.maximum.accumulate(prices)
    drawdowns = (prices - cumulative_max) / cumulative_max * 100  # Percentage drawdown
    metrics["max_drawdown_pct"] = float(np.min(drawdowns))  # Most negative = worst drawdown
    
    # Also calculate max run-up (largest cumulative rise from a trough)
    cumulative_min = np.minimum.accumulate(prices)
    runups = (prices - cumulative_min) / cumulative_min * 100
    metrics["max_runup_pct"] = float(np.max(runups))
    
    return metrics


def aggregate_metrics(metrics_list: list[dict[str, float]], confidence_level: float = 0.95) -> dict[str, dict]:
    """
    Aggregate metrics across multiple runs with uncertainty estimates.
    
    Args:
        metrics_list: List of metric dictionaries from each run
        confidence_level: Confidence level for CI (default 0.95 for 95% CI)
    
    Returns:
        Dictionary with aggregated statistics for each metric:
        {metric_name: {"mean": x, "std": y, "se": z, "ci_lower": a, "ci_upper": b, "n": n}}
    """
    if not metrics_list:
        return {}
    
    # Get all metric names
    metric_names = metrics_list[0].keys()
    
    aggregated = {}
    
    # Skip non-numeric metrics like 'timestamp'
    skip_metrics = {"timestamp"}
    
    for metric in metric_names:
        if metric in skip_metrics:
            continue
            
        values = [m[metric] for m in metrics_list if metric in m]
        n = len(values)
        
        if n == 0:
            continue
        
        # Skip if values are not numeric
        if not all(isinstance(v, (int, float)) for v in values):
            continue
            
        mean = np.mean(values)
        std = np.std(values, ddof=1) if n > 1 else 0.0  # Sample std
        se = std / np.sqrt(n) if n > 1 else 0.0  # Standard error
        
        # Confidence interval using t-distribution
        if n > 1:
            t_critical = stats.t.ppf((1 + confidence_level) / 2, df=n-1)
            ci_margin = t_critical * se
        else:
            ci_margin = 0.0
        
        aggregated[metric] = {
            "mean": float(mean),
            "std": float(std),
            "se": float(se),
            "ci_lower": float(mean - ci_margin),
            "ci_upper": float(mean + ci_margin),
            "n": n,
            "values": values  # Keep individual values for potential further analysis
        }
    
    return aggregated


def load_and_aggregate_config(config_name: str, config: dict) -> dict[str, Any]:
    """
    Load all runs for a configuration and aggregate metrics.
    
    Args:
        config_name: e.g., "baseline_var5"
        config: The full config dictionary
    
    Returns:
        Dictionary with configuration info and aggregated metrics
    """
    config_data = config["configurations"][config_name]
    timestamps = config_data["runs"]
    confidence_level = config["analysis_settings"]["confidence_level"]
    
    # Load all runs and calculate metrics
    metrics_list = []
    rounds_data_list = []  # Keep raw data for time series plots
    
    for timestamp in timestamps:
        data = load_single_run(config_name, timestamp)
        if data["rounds"] is not None:
            metrics = calculate_run_metrics(data["rounds"])
            metrics["timestamp"] = timestamp
            metrics_list.append(metrics)
            rounds_data_list.append(data["rounds"])
    
    if not metrics_list:
        print(f"Warning: No valid runs found for {config_name}")
        return None
    
    # Aggregate metrics
    aggregated = aggregate_metrics(metrics_list, confidence_level)
    
    return {
        "config_name": config_name,
        "condition": config_data["condition"],
        "variance": config_data["variance"],
        "description": config_data["description"],
        "n_runs": len(metrics_list),
        "timestamps": timestamps,
        "aggregated_metrics": aggregated,
        "per_run_metrics": metrics_list,
        "rounds_data": rounds_data_list  # Raw DataFrames for time series
    }


def load_all_configurations(config: dict = None) -> dict[str, Any]:
    """
    Load and aggregate all configurations from the config file.
    
    Returns:
        Dictionary mapping config_name to aggregated results
    """
    if config is None:
        config = load_config()
    
    results = {}
    
    for config_name in config["configurations"]:
        print(f"Loading {config_name}...")
        result = load_and_aggregate_config(config_name, config)
        if result is not None:
            results[config_name] = result
    
    return results


def create_summary_dataframe(results: dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary DataFrame comparing all configurations.
    
    Returns:
        DataFrame with one row per configuration, columns for each metric with CI
    """
    rows = []
    
    for config_name, data in results.items():
        row = {
            "Configuration": config_name,
            "Condition": data["condition"],
            "Variance (σ)": data["variance"],
            "N Runs": data["n_runs"],
        }
        
        # Add key metrics with CI
        metrics_to_include = [
            ("volatility", "Volatility (%)"),
            ("mean_herding_index", "Mean Herding Index"),
            ("max_herding_index", "Max Herding Index"),
            ("total_flash_events", "Flash Events"),
            ("mean_rel_deviation_pct", "Mean Price Deviation (%)"),
            ("rmse", "RMSE"),
        ]
        
        for metric_key, display_name in metrics_to_include:
            if metric_key in data["aggregated_metrics"]:
                m = data["aggregated_metrics"][metric_key]
                row[display_name] = m["mean"]
                row[f"{display_name} CI"] = f"[{m['ci_lower']:.2f}, {m['ci_upper']:.2f}]"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Sort by condition then variance
    df = df.sort_values(["Condition", "Variance (σ)"])
    
    return df


def print_summary(results: dict[str, Any]) -> None:
    """Print a formatted summary of results."""
    print("\n" + "="*80)
    print("AGGREGATED SIMULATION RESULTS")
    print("="*80)
    
    for config_name, data in sorted(results.items()):
        print(f"\n{config_name} ({data['n_runs']} runs)")
        print("-" * 40)
        
        metrics = data["aggregated_metrics"]
        
        key_metrics = [
            ("volatility", "Volatility", "%"),
            ("mean_herding_index", "Mean Herding Index", ""),
            ("total_flash_events", "Flash Events", ""),
            ("mean_rel_deviation_pct", "Price Deviation", "%"),
        ]
        
        for key, name, unit in key_metrics:
            if key in metrics:
                m = metrics[key]
                print(f"  {name}: {m['mean']:.3f}{unit} "
                      f"(95% CI: [{m['ci_lower']:.3f}, {m['ci_upper']:.3f}])")


# Main execution
if __name__ == "__main__":
    print("Loading configuration...")
    config = load_config()
    
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Confidence level: {config['analysis_settings']['confidence_level']*100}%")
    
    results = load_all_configurations(config)
    
    print_summary(results)
    
    # Create and display summary table
    summary_df = create_summary_dataframe(results)
    print("\n\nSUMMARY TABLE:")
    print(summary_df.to_string(index=False))
