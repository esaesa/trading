# data_loader.py
import pandas as pd
import warnings
import os
from logger_config import logger


def load_data(file_path, start_date=None, end_date=None, resample_freq=None):
    """Load historical price data from CSV."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    logger.info(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path, parse_dates=["timestamp"])
    data.set_index("timestamp", inplace=True)

    # Rename columns to standard format
    data = data.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    # Validate required columns
    required_columns = {"Open", "High", "Low", "Close", "Volume"}
    if not required_columns.issubset(data.columns):
        missing_cols = required_columns - set(data.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for duplicate indices
    if data.index.has_duplicates:
        warnings.warn("Duplicate timestamps found in the data. Dropping duplicates...")
        data = data[~data.index.duplicated(keep='first')]

    # Check if the index is sorted
    if not data.index.is_monotonic_increasing:
        warnings.warn("DatetimeIndex is not sorted. Sorting the data...")
        data.sort_index(inplace=True)

    # Handle missing values
    if data.isnull().values.any():
        warnings.warn("Data contains missing values. Filling NaNs with forward fill...")
        data = data.ffill().bfill()

    # Resample data if requested
    if resample_freq:
        logger.info(f"Resampling data to {resample_freq} frequency...")
        data = data.resample(resample_freq).last().ffill()

    # Filter data by date range
    if start_date:
        data = data.loc[start_date:]
    if end_date:
        data = data.loc[:end_date]



    return data