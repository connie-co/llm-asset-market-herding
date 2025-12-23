from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tslearn as ts

from tslearn.piecewise import PiecewiseAggregateApproximation

"Lets import the data first, then select the best window, about 30-50 steps to make" 
"the simulation reasonable"

bst = pd.read_csv(r'C:\Users\Constanza\CascadeProjects\jobsearch\CascadeProjects\windsurf-project\job_search\PhD positions\UCL-Econ\HistoricalData_BST_PRICE.csv')
nav= pd.read_csv(r'C:\Users\Constanza\CascadeProjects\jobsearch\CascadeProjects\windsurf-project\job_search\PhD positions\UCL-Econ\HistoricalData_XBSTX_NAV.csv')

bst=bst.iloc[:,[0,1]]
nav=nav.iloc[:,[0,1]]

# --- Simple Plot of Raw BST Price (before any conversions) ---
print("Raw BST data:")
print(bst.head())

# Convert price column to numeric (remove $ sign) for plotting
bst_plot = bst.copy()
bst_plot['Close/Last'] = bst_plot['Close/Last'].str.replace('$', '', regex=False).astype(float)

fig_raw, ax_raw = plt.subplots(figsize=(12, 5))
ax_raw.plot(range(len(bst_plot)), bst_plot['Close/Last'], color='blue', linewidth=1)
ax_raw.set_title('Raw BST Price (before conversions/merging)')
ax_raw.set_xlabel('Index (most recent first)')
ax_raw.set_ylabel('Price ($)')
ax_raw.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Data Cleaning and Merging ---
# Convert 'Date' column to datetime objects
bst['Date'] = pd.to_datetime(bst['Date'])
nav['Date'] = pd.to_datetime(nav['Date'])

bst.head()
# Set 'Date' as the index for both dataframes
bst.set_index('Date', inplace=True)
nav.set_index('Date', inplace=True)

# Rename columns for clarity before merging
bst.rename(columns={'Close/Last': 'Price'}, inplace=True)
#bst['Price'] = bst['Price'].str.replace('$', '', regex=False)
nav.rename(columns={'Close/Last': 'NAV'}, inplace=True)
#nav['NAV'] = nav['NAV'].str.replace('$', '', regex=False)

# Merge the two dataframes on the date index
# Use an inner join to only keep dates where both price and NAV data exist
df = bst.join(nav, how='inner')
#df[['Price', 'NAV']] = df[['Price', 'NAV']].str.replace('$', '', regex=False)
# Drop any rows with missing values that might have slipped through
df.dropna(inplace=True)
df = df.replace({'\$': ''}, regex=True)

# Convert Price and NAV to numeric for calculations
df['Price'] = pd.to_numeric(df['Price'])
df['NAV'] = pd.to_numeric(df['NAV'])

# Calculate Discount/Premium as percentage of NAV
# Discount is negative (Price < NAV), Premium is positive (Price > NAV)
df['Discount_Premium_Pct'] = ((df['Price'] - df['NAV']) / df['NAV']) * 100

print("Merged and cleaned data:")
print(df.head())
print(f"\nData spans from {df.index.min().date()} to {df.index.max().date()} with {len(df)} trading days.")

# --- Plot 1: Price vs NAV, Plot 2: Discount/Premium ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# Plot 1: Price and NAV
ax1.plot(df.index, df['Price'], label='Market Price', color='blue', linewidth=1.5)
ax1.plot(df.index, df['NAV'], label='NAV (True Value)', color='green', linewidth=1.5, linestyle='--')
ax1.set_title('BST: Market Price vs. Net Asset Value (NAV)')
ax1.set_ylabel('Value ($)')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Plot 2: Discount/Premium as percentage of NAV
ax2.plot(df.index, df['Discount_Premium_Pct'], label='Discount/Premium (%)', color='purple', linewidth=1.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
ax2.fill_between(df.index, 0, df['Discount_Premium_Pct'], 
                  where=(df['Discount_Premium_Pct'] >= 0), color='green', alpha=0.3, label='Premium')
ax2.fill_between(df.index, 0, df['Discount_Premium_Pct'], 
                  where=(df['Discount_Premium_Pct'] < 0), color='red', alpha=0.3, label='Discount')
ax2.set_title('BST: Discount/Premium as Percentage of NAV')
ax2.set_xlabel('Date')
ax2.set_ylabel('Discount/Premium (%)')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Save the plot
save_path = Path(__file__).parent.parent / 'data' / 'plots' / 'BST_price_nav_discount.png'
save_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(save_path, dpi=150, bbox_inches='tight', format='png')
print(f"\nPlot saved to: {save_path}")

plt.show()