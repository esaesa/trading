# data_loader.py
import pandas as pd
import warnings
import os
from logger_config import logger


def load_data(file_path, start_date=None, end_date=None, resample_freq=None):
    """
    Load OHLCV data from Feather, filter by [start_date, end_date], 
    rename columns, set timestamp index, and optionally resample.
    """
    # ★ always target .feather
    feather_path = file_path.replace('.csv', '.feather')
    if not os.path.exists(feather_path):
        raise FileNotFoundError(f"Feather file not found: {feather_path}")
    logger.info(f"Loading data from {feather_path}...")

    # ★ direct Feather load
    df = pd.read_feather(feather_path)
    
    # ★ filter BEFORE indexing
    if start_date:
        df = df[df['timestamp'] >= start_date]
    if end_date:
        df = df[df['timestamp'] <= end_date]

    # ★ then set index
    df.set_index('timestamp', inplace=True)

    # Rename columns to standard format
    df = df.rename(
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
    if not required_columns.issubset(df.columns):
        missing_cols = required_columns - set(df.columns)
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for duplicate indices
    if df.index.has_duplicates:
        warnings.warn("Duplicate timestamps found in the data. Dropping duplicates...")
        df = df[~df.index.duplicated(keep='first')]

    # Check if the index is sorted
    if not df.index.is_monotonic_increasing:
        warnings.warn("DatetimeIndex is not sorted. Sorting the data...")
        df.sort_index(inplace=True)

    # Handle missing values
    if df.isnull().values.any():
        warnings.warn("Data contains missing values. Filling NaNs with forward fill...")
        df = df.ffill().bfill()

    # Resample data if requested. As it is prices we need the last value of the period.
    if resample_freq:
        logger.info(f"Resampling data to {resample_freq} frequency...")
        df = df.resample(resample_freq).last().ffill()

    # Filter data by date range
    """     if start_date:
            df = df.loc[start_date:]
        if end_date:
            df = df.loc[:end_date] """



    return df