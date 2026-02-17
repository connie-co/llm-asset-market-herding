"""Check if runs are identical (same seed issue)."""
import pandas as pd
import numpy as np
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

# Check baseline_var5 runs
runs = [
    "20251228_035158",
    "20251228_181136",
    "20251228_230314",
    "20251229_142546",
]

print("CHECKING IF RUNS ARE IDENTICAL (baseline_var5)")
print("="*60)

dfs = []
for ts in runs:
    path = RESULTS_DIR / f"baseline_var5_rounds_{ts}.csv"
    if path.exists():
        df = pd.read_csv(path)
        dfs.append((ts, df))
        print(f"\n{ts}:")
        print(f"  First 5 returns: {df['return_pct'].iloc[:5].tolist()}")
        print(f"  Sum of all returns: {df['return_pct'].sum():.4f}")
        print(f"  Final price: ${df['price_after'].iloc[-1]:.2f}")

print("\n" + "="*60)
print("COMPARISON:")
if len(dfs) >= 2:
    df1 = dfs[0][1]
    df2 = dfs[1][1]
    
    # Check if returns are identical
    returns_identical = np.allclose(df1['return_pct'].values, df2['return_pct'].values)
    prices_identical = np.allclose(df1['price_after'].values, df2['price_after'].values)
    
    print(f"Returns identical between run 1 and 2: {returns_identical}")
    print(f"Prices identical between run 1 and 2: {prices_identical}")
    
    if not returns_identical:
        diff = df1['return_pct'].values - df2['return_pct'].values
        print(f"Max difference in returns: {np.max(np.abs(diff)):.6f}%")
