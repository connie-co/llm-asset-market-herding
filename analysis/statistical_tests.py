"""
Statistical tests for comparing baseline vs herding conditions.

Performs hypothesis tests to determine if differences between conditions
are statistically significant.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Any

from .aggregate_runs import load_config, load_all_configurations


def welch_t_test(values1: list[float], values2: list[float]) -> dict:
    """
    Perform Welch's t-test (unequal variances t-test).
    
    Args:
        values1: Sample 1 values (e.g., baseline)
        values2: Sample 2 values (e.g., herding)
    
    Returns:
        Dictionary with test results
    """
    n1, n2 = len(values1), len(values2)
    
    if n1 < 2 or n2 < 2:
        return {
            "t_statistic": np.nan,
            "p_value": np.nan,
            "df": np.nan,
            "significant_05": False,
            "significant_01": False,
            "error": "Insufficient samples (need at least 2 per group)"
        }
    
    t_stat, p_value = stats.ttest_ind(values1, values2, equal_var=False)
    
    # Calculate degrees of freedom (Welch-Satterthwaite)
    var1, var2 = np.var(values1, ddof=1), np.var(values2, ddof=1)
    df = ((var1/n1 + var2/n2)**2) / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))
    
    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "df": float(df),
        "significant_05": p_value < 0.05,
        "significant_01": p_value < 0.01,
        "n1": n1,
        "n2": n2,
        "mean1": float(np.mean(values1)),
        "mean2": float(np.mean(values2)),
        "difference": float(np.mean(values2) - np.mean(values1)),
    }


def cohens_d(values1: list[float], values2: list[float]) -> float:
    """
    Calculate Cohen's d effect size.
    
    Positive d means values2 > values1.
    """
    n1, n2 = len(values1), len(values2)
    
    if n1 < 2 or n2 < 2:
        return np.nan
    
    mean1, mean2 = np.mean(values1), np.mean(values2)
    var1, var2 = np.var(values1, ddof=1), np.var(values2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    return float((mean2 - mean1) / pooled_std)


def interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    if np.isnan(d):
        return "N/A"
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def compare_conditions_for_metric(
    results: dict[str, Any],
    metric_key: str,
    variance: int
) -> dict:
    """
    Compare baseline vs herding for a specific metric and variance level.
    
    Returns:
        Dictionary with test results and effect sizes
    """
    baseline_key = f"baseline_var{variance}"
    herding_key = f"herding_var{variance}"
    
    if baseline_key not in results or herding_key not in results:
        return {"error": "Missing configuration"}
    
    if metric_key not in results[baseline_key]["aggregated_metrics"]:
        return {"error": f"Metric {metric_key} not found"}
    
    baseline_values = results[baseline_key]["aggregated_metrics"][metric_key]["values"]
    herding_values = results[herding_key]["aggregated_metrics"][metric_key]["values"]
    
    # T-test
    t_test_results = welch_t_test(baseline_values, herding_values)
    
    # Effect size
    d = cohens_d(baseline_values, herding_values)
    
    return {
        "metric": metric_key,
        "variance": variance,
        "baseline_mean": t_test_results.get("mean1", np.nan),
        "herding_mean": t_test_results.get("mean2", np.nan),
        "difference": t_test_results.get("difference", np.nan),
        "t_statistic": t_test_results["t_statistic"],
        "p_value": t_test_results["p_value"],
        "significant_05": t_test_results["significant_05"],
        "significant_01": t_test_results["significant_01"],
        "cohens_d": d,
        "effect_interpretation": interpret_cohens_d(d),
        "n_baseline": t_test_results.get("n1", 0),
        "n_herding": t_test_results.get("n2", 0),
    }


def run_all_comparisons(results: dict[str, Any]) -> pd.DataFrame:
    """
    Run statistical comparisons for all metrics and variance levels.
    
    Returns:
        DataFrame with all comparison results
    """
    metrics = [
        "volatility",
        "mean_herding_index",
        "max_herding_index",
        "total_flash_events",
        "mean_rel_deviation_pct",
        "rmse",
        "max_drawdown_pct",
        "max_runup_pct",
    ]
    
    variances = [5, 12, 20]
    
    rows = []
    for metric in metrics:
        for var in variances:
            comparison = compare_conditions_for_metric(results, metric, var)
            if "error" not in comparison:
                rows.append(comparison)
    
    return pd.DataFrame(rows)


def format_p_value(p: float) -> str:
    """Format p-value for display."""
    if np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "<0.001***"
    elif p < 0.01:
        return f"{p:.3f}**"
    elif p < 0.05:
        return f"{p:.3f}*"
    else:
        return f"{p:.3f}"


def print_comparison_report(results: dict[str, Any]) -> None:
    """Print a formatted statistical comparison report."""
    comparisons_df = run_all_comparisons(results)
    
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON: BASELINE vs HERDING")
    print("="*80)
    print("H0: No difference between conditions")
    print("H1: Herding condition differs from baseline")
    print("Test: Welch's t-test (unequal variances)")
    print("-"*80)
    
    for metric in comparisons_df["metric"].unique():
        metric_df = comparisons_df[comparisons_df["metric"] == metric]
        
        print(f"\n{metric.upper().replace('_', ' ')}")
        print("-" * 40)
        
        for _, row in metric_df.iterrows():
            sig_marker = ""
            if row["significant_01"]:
                sig_marker = "**"
            elif row["significant_05"]:
                sig_marker = "*"
            
            print(f"  σ={row['variance']:2d}: "
                  f"Baseline={row['baseline_mean']:.3f}, "
                  f"Herding={row['herding_mean']:.3f}, "
                  f"Δ={row['difference']:+.3f}, "
                  f"p={format_p_value(row['p_value'])}, "
                  f"d={row['cohens_d']:.2f} ({row['effect_interpretation']}){sig_marker}")
    
    print("\n" + "-"*80)
    print("* p < 0.05, ** p < 0.01, *** p < 0.001")
    print("Cohen's d: |d|<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large")


def export_results_to_csv(results: dict[str, Any], output_path: str = None) -> pd.DataFrame:
    """
    Export statistical comparison results to CSV.
    
    Returns:
        The comparison DataFrame
    """
    from pathlib import Path
    
    comparisons_df = run_all_comparisons(results)
    
    if output_path is None:
        output_path = Path(__file__).parent.parent / "plots" / "statistical_comparisons.csv"
    
    comparisons_df.to_csv(output_path, index=False)
    print(f"\nStatistical comparisons exported to: {output_path}")
    
    return comparisons_df


# Main execution
if __name__ == "__main__":
    print("Loading configuration and data...")
    config = load_config()
    results = load_all_configurations(config)
    
    print_comparison_report(results)
    
    # Export to CSV
    df = export_results_to_csv(results)
    
    print("\n\nSUMMARY TABLE:")
    print(df.to_string(index=False))
