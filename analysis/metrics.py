"""Metrics for analyzing simulation results."""

import numpy as np
import pandas as pd


def calculate_volatility(prices: list[float] | pd.Series) -> float:
    """
    Calculate price volatility as standard deviation of returns.

    Args:
        prices: List or Series of prices

    Returns:
        Volatility (std of percentage returns)
    """
    prices = np.array(prices)
    if len(prices) < 2:
        return 0.0

    returns = np.diff(prices) / prices[:-1] * 100
    return float(np.std(returns))


def calculate_returns(prices: list[float] | pd.Series) -> np.ndarray:
    """
    Calculate percentage returns from prices.

    Args:
        prices: List or Series of prices

    Returns:
        Array of percentage returns
    """
    prices = np.array(prices)
    if len(prices) < 2:
        return np.array([])

    return np.diff(prices) / prices[:-1] * 100


def count_flash_events(
    returns: np.ndarray, threshold: float = 5.0
) -> dict[str, int]:
    """
    Count flash crash and flash rally events.

    Args:
        returns: Array of percentage returns
        threshold: Percentage threshold for flash events

    Returns:
        Dictionary with crash and rally counts
    """
    return {
        "flash_crashes": int(np.sum(returns < -threshold)),
        "flash_rallies": int(np.sum(returns > threshold)),
    }


def calculate_herding_index(actions_df: pd.DataFrame) -> pd.Series:
    """
    Calculate herding index per round.

    Herding index = |net_demand| / n_agents
    Ranges from 0 (no herding) to 1 (complete herding)

    Args:
        actions_df: DataFrame with 'n_buys', 'n_sells', 'n_holds' columns

    Returns:
        Series of herding indices
    """
    n_agents = actions_df["n_buys"] + actions_df["n_sells"] + actions_df["n_holds"]
    net_demand = actions_df["n_buys"] - actions_df["n_sells"]
    return np.abs(net_demand) / n_agents


def calculate_price_efficiency(
    prices: list[float] | pd.Series,
    true_values: list[float] | pd.Series,
) -> dict[str, float]:
    """
    Calculate price efficiency metrics.

    Args:
        prices: List or Series of prices
        true_values: List or Series of true values

    Returns:
        Dictionary with efficiency metrics
    """
    prices = np.array(prices)
    true_values = np.array(true_values)

    deviations = np.abs(prices - true_values)
    relative_deviations = deviations / true_values * 100

    return {
        "mean_absolute_deviation": float(np.mean(deviations)),
        "max_absolute_deviation": float(np.max(deviations)),
        "mean_relative_deviation_pct": float(np.mean(relative_deviations)),
        "rmse": float(np.sqrt(np.mean((prices - true_values) ** 2))),
    }


def compare_experiments(
    baseline_df: pd.DataFrame,
    herding_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare metrics between baseline and herding experiments.

    Args:
        baseline_df: Results from baseline (diverse info) experiment
        herding_df: Results from herding (homogeneous info) experiment

    Returns:
        DataFrame comparing key metrics
    """
    metrics = []

    for name, df in [("Baseline", baseline_df), ("Herding", herding_df)]:
        returns = calculate_returns(df["price_after"].values)
        flash_events = count_flash_events(returns)

        metrics.append({
            "Experiment": name,
            "Volatility (%)": calculate_volatility(df["price_after"].values),
            "Mean Return (%)": float(np.mean(returns)) if len(returns) > 0 else 0,
            "Flash Crashes": flash_events["flash_crashes"],
            "Flash Rallies": flash_events["flash_rallies"],
            "Mean Herding Index": float(df["herding_index"].mean()),
            "Max Herding Index": float(df["herding_index"].max()),
        })

    return pd.DataFrame(metrics)
