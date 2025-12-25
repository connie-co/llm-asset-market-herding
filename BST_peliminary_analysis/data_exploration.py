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

# --- Data Cleaning and Merging ---
# Convert 'Date' column to datetime objects
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

# Sort by date ascending so Step 0 = earliest date (2020), Step N = latest (2025)
df.sort_index(ascending=True, inplace=True)
df = df.replace({'\$': ''}, regex=True)

# Convert Price and NAV to numeric for calculations
df['Price'] = pd.to_numeric(df['Price'])
df['NAV'] = pd.to_numeric(df['NAV'])
# --- Piecewise Aggregate Approximation (PAA) ---
# Define the desired length of the time series for the simulation
N_STEPS = 200

# Extract the time series as numpy arrays
price_series = df['Price'].values
nav_series = df['NAV'].values

# Reshape for tslearn which expects (n_samples, n_timesteps, n_features)
# Here we have 1 sample, len(df) timesteps, and 1 feature
price_series_reshaped = price_series.reshape(1, -1)
nav_series_reshaped = nav_series.reshape(1, -1)
print('notice!!!', nav_series_reshaped.shape)


# Initialize the PAA model


# a ver otra v3z
paa = PiecewiseAggregateApproximation(n_segments=3)
data = [[-1., 2., 0.1, -1., 1., -1.]]
paa_data = paa.fit_transform(data)


print('numeric test',paa_data)

paa = PiecewiseAggregateApproximation(n_segments=N_STEPS)

# Fit and transform the data
price_paa = paa.fit_transform(price_series_reshaped)
nav_paa = paa.fit_transform(nav_series_reshaped)
#print(f"PAA NAV values (first 5): {nav_paa[:5]}")
# The output is 3D: (n_samples, n_segments, n_features)
# We have 1 sample, N_STEPS segments, 1 feature
# Extract the first (and only) sample and flatten to 1D

price_paa_squeezed = price_paa.flatten()
nav_paa_squeezed = nav_paa.flatten()

print(f"\nApplied PAA to reduce series from {len(df)} to {N_STEPS} steps.")
print(f"PAA output shape: {price_paa.shape}")
#print(f"Squeezed PAA shape: {price_paa_squeezed.shape}")

# --- Visualization ---
# Create a date index for PAA data by selecting N_STEPS equally spaced dates from original data
# This allows us to plot both original and PAA data on the same date axis
total_days = len(df)
indices_to_sample = np.linspace(0, total_days - 1, N_STEPS, dtype=int)
paa_date_index = df.index[indices_to_sample]

print(f"\nPAA Price values (first 5): {price_paa_squeezed[:5]}")
print(f"PAA NAV values (first 5): {nav_paa_squeezed[:5]}")
print(f"PAA date index (first 5): {paa_date_index[:5]}")
print(f"PAA date index (last 5): {paa_date_index[-5:]}")

fig, ax = plt.subplots(figsize=(15, 7))

# Plot original data (full resolution) with transparency
ax.plot(df.index, df['Price'], label='Original Price (full data)', 
        color='skyblue', alpha=0.4, linewidth=1)
ax.plot(df.index, df['NAV'], label='Original NAV (full data)', 
        color='lightcoral', alpha=0.4, linewidth=1, linestyle='--')

# Plot PAA data (downsampled to N_STEPS) with dates
ax.plot(paa_date_index, price_paa_squeezed, label=f'PAA Price ({N_STEPS} steps)', 
        color='blue', linewidth=2, marker='o', markersize=3)
ax.plot(paa_date_index, nav_paa_squeezed, label=f'PAA NAV ({N_STEPS} steps)', 
        color='red', linewidth=2, linestyle='--', marker='x', markersize=3)

ax.set_title('BST Price vs. NAV: Original vs. PAA Downsampled')
ax.set_xlabel('Date')
ax.set_ylabel('Value ($)')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# --- Add secondary x-axis for step numbers ---
ax2 = ax.twiny()  # Create twin axis sharing y-axis
ax2.set_xlim(ax.get_xlim())  # Match the x-limits

# Set step ticks at regular intervals (e.g., every 25 steps for 200 total)
step_interval = max(1, N_STEPS // 8)  # Show ~8 tick marks
step_ticks = list(range(0, N_STEPS, step_interval)) + [N_STEPS - 1]
# Map step numbers to corresponding dates for positioning
step_tick_positions = [paa_date_index[i] for i in step_ticks]
ax2.set_xticks(step_tick_positions)
ax2.set_xticklabels([str(s) for s in step_ticks])
ax2.set_xlabel('Step (PAA segment)')

plt.tight_layout()
plt.show()


