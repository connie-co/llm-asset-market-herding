"""Debug script to analyze volatility calculation."""
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

files = [
    ('baseline_var5', '20251228_035158'),
    ('baseline_var12', '20251228_072804'),
    ('baseline_var20', '20251228_111552'),
]

print('DETAILED RETURNS ANALYSIS')
print('='*70)

for config, ts in files:
    df = pd.read_csv(RESULTS_DIR / f"{config}_rounds_{ts}.csv")
    returns = df['return_pct'].values
    
    print(f'\n{config}:')
    print(f'  Number of returns: {len(returns)}')
    print(f'  Mean return: {np.mean(returns):.4f}%')
    print(f'  Sum of returns: {np.sum(returns):.4f}%')
    print(f'  Std (volatility): {np.std(returns):.4f}%')
    print(f'  Min return: {np.min(returns):.4f}%')
    print(f'  Max return: {np.max(returns):.4f}%')
    print(f'  Mean of |returns|: {np.mean(np.abs(returns)):.4f}%')
    
    # Count positive vs negative returns
    n_positive = np.sum(returns > 0)
    n_negative = np.sum(returns < 0)
    n_zero = np.sum(returns == 0)
    print(f'  Positive returns: {n_positive}, Negative: {n_negative}, Zero: {n_zero}')
    
    # Show first 10 returns
    print(f'  First 10 returns: {[round(r, 2) for r in returns[:10]]}')

print('\n' + '='*70)
print('VERIFICATION: Does std handle positive/negative correctly?')
print('='*70)
test_returns = np.array([+4, -4, +4, -4, +4, -4])
print(f'Test returns: {test_returns}')
print(f'  Mean: {np.mean(test_returns):.2f} (should be 0)')
print(f'  Sum: {np.sum(test_returns):.2f} (should be 0)')
print(f'  Std: {np.std(test_returns):.2f} (should be 4.0)')
print(f'  Mean of |returns|: {np.mean(np.abs(test_returns)):.2f} (should be 4.0)')
