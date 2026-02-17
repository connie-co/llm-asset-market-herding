"""
Deep dive into volatility analysis.
Plots volatility across noise levels and detailed return analysis with 3-sigma bands.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import os

# Setup paths
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
OUTPUT_DIR = Path(__file__).parent.parent / "plots" / "volatility_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')

def get_latest_run(config_name: str) -> tuple[str, pd.DataFrame] | None:
    """Find and load the latest run for a configuration."""
    pattern = str(RESULTS_DIR / f"{config_name}_rounds_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found for {config_name}")
        return None
        
    # Sort by modification time (latest first)
    latest_file = max(files, key=os.path.getmtime)
    timestamp = latest_file.split("_")[-1].replace(".csv", "")
    
    print(f"Loaded latest {config_name}: {timestamp}")
    return timestamp, pd.read_csv(latest_file)

def calculate_metrics(df: pd.DataFrame):
    """Calculate returns and volatility."""
    prices = df['price_after'].values
    returns = np.diff(prices) / prices[:-1] * 100
    volatility = np.std(returns)
    return prices, returns, volatility

def plot_volatility_comparison(configs_data: dict):
    """Plot volatility vs noise level."""
    noises = []
    vols = []
    labels = []
    
    # Extract noise level from config name (e.g., baseline_var5 -> 5)
    sorted_configs = sorted(configs_data.keys(), key=lambda x: int(x.split('var')[1]))
    
    for config in sorted_configs:
        _, _, vol = configs_data[config]
        noise_level = int(config.split('var')[1])
        
        noises.append(noise_level)
        vols.append(vol)
        labels.append(f"σ={noise_level}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, vols, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_title("Price Volatility vs. Signal Noise Level", fontsize=14, fontweight='bold')
    ax.set_ylabel("Volatility (Std Dev of Returns %)", fontsize=12)
    ax.set_xlabel("Signal Noise (σ)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add suspicion note
    if vols[0] > vols[-1]:
        txt = "Note: Lower noise leading to higher volatility\nsuggests agent coordination/herding."
        ax.text(0.95, 0.95, txt, transform=ax.transAxes, 
                ha='right', va='top', bbox=dict(facecolor='white', alpha=0.8))
        
    plt.tight_layout()
    save_path = OUTPUT_DIR / "volatility_vs_noise.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def plot_detailed_run(config_name: str, timestamp: str, df: pd.DataFrame):
    """Plot Price and Returns with 3-sigma bands."""
    prices, returns, vol = calculate_metrics(df)
    rounds_prices = df['round'].values
    rounds_returns = rounds_prices[1:]  # Returns are one shorter
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Price Plot
    ax1.plot(rounds_prices, prices, label='Market Price', color='blue', linewidth=2)
    ax1.plot(rounds_prices, df['true_value'], label='True Value (NAV)', color='black', linestyle='--', alpha=0.6)
    ax1.set_title(f"Detailed Analysis: {config_name} (Run {timestamp})", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Price ($)", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Returns Plot
    ax2.plot(rounds_returns, returns, label='Returns', color='purple', marker='o', markersize=4, alpha=0.7)
    
    # Add +/- 3 Sigma bands
    mean_ret = np.mean(returns)
    upper_3std = mean_ret + 3 * vol
    lower_3std = mean_ret - 3 * vol
    
    ax2.axhline(y=mean_ret, color='black', linestyle='-', alpha=0.3, label='Mean')
    ax2.axhline(y=upper_3std, color='red', linestyle='--', label=f'+3σ ({upper_3std:.2f}%)')
    ax2.axhline(y=lower_3std, color='red', linestyle='--', label=f'-3σ ({lower_3std:.2f}%)')
    
    # Fill "normal" zone
    ax2.fill_between(rounds_returns, lower_3std, upper_3std, color='green', alpha=0.05, label='Normal Range (±3σ)')
    
    ax2.set_ylabel("Return (%)", fontsize=12)
    ax2.set_xlabel("Round", fontsize=12)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Annotate Volatility
    ax2.text(0.02, 0.9, f"Volatility (σ) = {vol:.2f}%", transform=ax2.transAxes, 
             fontweight='bold', bbox=dict(facecolor='white', edgecolor='purple', alpha=0.8))

    plt.tight_layout()
    save_path = OUTPUT_DIR / f"detailed_{config_name}.png"
    plt.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    plt.close()

def main():
    configs = ['baseline_var5', 'baseline_var12', 'baseline_var20']
    
    configs_data = {}
    
    print("Loading data...")
    for config in configs:
        result = get_latest_run(config)
        if result:
            ts, df = result
            configs_data[config] = calculate_metrics(df)
            # Plot individual run detail
            plot_detailed_run(config, ts, df)
    
    if configs_data:
        # Plot comparison
        plot_volatility_comparison(configs_data)

if __name__ == "__main__":
    main()
