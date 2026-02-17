"""Load and prepare real-world CEF data for simulation."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tslearn.piecewise import PiecewiseAggregateApproximation


def load_bst_nav_data(paa_steps: int = 100) -> np.ndarray:
    """
    Load BST CEF NAV data and downsample using PAA.
    
    Args:
        paa_steps: Number of PAA segments to downsample to (default: 200)
                   This controls the level of detail preserved from the original data.
        
    Returns:
        1D numpy array of NAV values downsampled to paa_steps
        
    Raises:
        FileNotFoundError: If the BST data files are not found
    """
    # Get the data directory from environment variable or use default relative path
    data_dir = os.getenv('CEF_DATA_DIR')
    
    if data_dir:
        # Use custom data directory from environment variable
        project_root = Path(data_dir)
    else:
        # Default: assume data is in project root (parent of llm-asset-market-herding/)
        project_root = Path(__file__).parent.parent.parent
    
    # Define file paths
    bst_price_path = project_root / 'HistoricalData_BST_PRICE.csv'
    nav_path = project_root / 'HistoricalData_XBSTX_NAV.csv'
    
    # Check if files exist
    if not bst_price_path.exists():
        raise FileNotFoundError(
            f"BST price data not found at {bst_price_path}. "
            "Please download the data from Yahoo Finance."
        )
    if not nav_path.exists():
        raise FileNotFoundError(
            f"NAV data not found at {nav_path}. "
            "Please download the data from Yahoo Finance."
        )
    
    # Load data
    bst = pd.read_csv(bst_price_path)
    nav = pd.read_csv(nav_path)
    
    # Select only Date and Close/Last columns
    bst = bst.iloc[:, [0, 1]]
    nav = nav.iloc[:, [0, 1]]
    
    # Convert Date to datetime
    bst['Date'] = pd.to_datetime(bst['Date'])
    nav['Date'] = pd.to_datetime(nav['Date'])
    
    # Set Date as index
    bst.set_index('Date', inplace=True)
    nav.set_index('Date', inplace=True)
    
    # Rename columns
    bst.rename(columns={'Close/Last': 'Price'}, inplace=True)
    nav.rename(columns={'Close/Last': 'NAV'}, inplace=True)
    
    # Merge on date (inner join to keep only matching dates)
    df = bst.join(nav, how='inner')
    df.dropna(inplace=True)
    
    # Sort by date ascending so Step 0 = earliest date (2020), Step N = latest (2025)
    df.sort_index(ascending=True, inplace=True)
    
    # Remove $ signs and convert to numeric
    df = df.replace({r'\$': ''}, regex=True)
    df['Price'] = pd.to_numeric(df['Price'])
    df['NAV'] = pd.to_numeric(df['NAV'])
    
    # Extract NAV series
    nav_series = df['NAV'].values
    
    # Reshape for tslearn: (n_samples, n_timesteps, n_features)
    # We have 1 sample, len(df) timesteps, 1 feature
    nav_series_reshaped = nav_series.reshape(1, -1)
    
    # Apply PAA to downsample
    paa = PiecewiseAggregateApproximation(n_segments=paa_steps)
    nav_paa = paa.fit_transform(nav_series_reshaped)
    
    # Flatten to 1D array
    nav_paa_squeezed = nav_paa.flatten()
    
    return nav_paa_squeezed


def load_bst_data(paa_steps: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """
    Load both BST NAV and Price data, downsampled using PAA.
    
    This function is useful for plotting comparisons between NAV (true value)
    and market price.
    
    Args:
        paa_steps: Number of PAA segments to downsample to (default: 200)
        
    Returns:
        Tuple of (nav_paa, price_paa) - both as 1D numpy arrays
        
    Raises:
        FileNotFoundError: If the BST data files are not found
    """
    # Get the data directory from environment variable or use default relative path
    data_dir = os.getenv('CEF_DATA_DIR')
    
    if data_dir:
        project_root = Path(data_dir)
    else:
        project_root = Path(__file__).parent.parent.parent
    
    # Define file paths
    bst_price_path = project_root / 'HistoricalData_BST_PRICE.csv'
    nav_path = project_root / 'HistoricalData_XBSTX_NAV.csv'
    
    # Check if files exist
    if not bst_price_path.exists():
        raise FileNotFoundError(f"BST price data not found at {bst_price_path}.")
    if not nav_path.exists():
        raise FileNotFoundError(f"NAV data not found at {nav_path}.")
    
    # Load data
    bst = pd.read_csv(bst_price_path)
    nav = pd.read_csv(nav_path)
    
    # Select only Date and Close/Last columns
    bst = bst.iloc[:, [0, 1]]
    nav = nav.iloc[:, [0, 1]]
    
    # Convert Date to datetime
    bst['Date'] = pd.to_datetime(bst['Date'])
    nav['Date'] = pd.to_datetime(nav['Date'])
    
    # Set Date as index
    bst.set_index('Date', inplace=True)
    nav.set_index('Date', inplace=True)
    
    # Rename columns
    bst.rename(columns={'Close/Last': 'Price'}, inplace=True)
    nav.rename(columns={'Close/Last': 'NAV'}, inplace=True)
    
    # Merge on date (inner join to keep only matching dates)
    df = bst.join(nav, how='inner')
    df.dropna(inplace=True)
    
    # Sort by date ascending so Step 0 = earliest date (2020), Step N = latest (2025)
    df.sort_index(ascending=True, inplace=True)
    
    # Remove $ signs and convert to numeric
    df = df.replace({r'\$': ''}, regex=True)
    df['Price'] = pd.to_numeric(df['Price'])
    df['NAV'] = pd.to_numeric(df['NAV'])
    
    # Extract both series
    nav_series = df['NAV'].values
    price_series = df['Price'].values
    
    # Reshape for tslearn
    nav_reshaped = nav_series.reshape(1, -1)
    price_reshaped = price_series.reshape(1, -1)
    
    # Apply PAA to downsample
    paa = PiecewiseAggregateApproximation(n_segments=paa_steps)
    nav_paa = paa.fit_transform(nav_reshaped)
    price_paa = paa.fit_transform(price_reshaped)
    
    # Flatten to 1D arrays
    nav_paa_squeezed = nav_paa.flatten()
    price_paa_squeezed = price_paa.flatten()
    
    return nav_paa_squeezed, price_paa_squeezed


def get_true_value_series(
    n_rounds: int,
    use_cef_data: bool = False,
    initial_value: float = 100.0,
    drift: float = 0.0,
    volatility: float = 2.0,
    random_seed: int | None = None,
    paa_steps: int = 200,
) -> list[float]:
    """
    Get true value series - either from CEF data or random walk.
    
    Args:
        n_rounds: Number of simulation rounds (will trim CEF data to this length if needed)
        use_cef_data: If True, use CEF NAV data; if False, use random walk
        initial_value: Initial value for random walk (ignored if use_cef_data=True)
        drift: Drift parameter for random walk (ignored if use_cef_data=True)
        volatility: Volatility for random walk (ignored if use_cef_data=True)
        random_seed: Random seed for reproducibility
        paa_steps: Number of PAA segments to use when loading CEF data (default: 200)
                   This preserves more detail from the original data.
                   If n_rounds < paa_steps, the data will be trimmed to n_rounds.
        
    Returns:
        List of true values for each time step
        
    Note:
        For CEF data: Always applies PAA with paa_steps (default 200) to preserve detail,
        then trims to n_rounds if you want a shorter simulation.
        For random walk: Generates exactly n_rounds values.
    """
    if use_cef_data:
        # Load CEF NAV data with PAA downsampling
        nav_data = load_bst_nav_data(paa_steps=paa_steps)
        
        # Trim to n_rounds if simulation is shorter than PAA data
        if n_rounds < len(nav_data):
            nav_data = nav_data[:n_rounds]
        elif n_rounds > len(nav_data):
            raise ValueError(
                f"n_rounds ({n_rounds}) exceeds available CEF data length ({len(nav_data)}). "
                f"Either reduce n_rounds or increase paa_steps."
            )
        
        return nav_data.tolist()
    else:
        # Generate random walk
        if random_seed is not None:
            np.random.seed(random_seed)
        
        true_values = [initial_value]
        for _ in range(n_rounds - 1):
            change = np.random.normal(drift, volatility)
            new_value = true_values[-1] + change
            true_values.append(new_value)
        
        return true_values

