# ===== data_loader.py =====
import pandas as pd
import warnings
import os
from logger_config import logger
from rich.console import Console


# [ADDED] small helper to standardize an empty OHLCV frame with a DatetimeIndex
def _empty_ohlcv_like(columns=None):
    """
    Return an empty OHLCV DataFrame with a DatetimeIndex and standard columns.
    """
    cols = ["Open", "High", "Low", "Close", "Volume"] if columns is None else columns
    empty = pd.DataFrame(columns=cols)
    empty.index = pd.DatetimeIndex([], name="timestamp")
    return empty

# [ADDED] safe parse helper: returns (timestamp_or_None, is_invalid_flag)
def _safe_parse_datetime(value, field_name: str):
    """
    Try to parse value to pandas.Timestamp.
    - Accepts None / "" as open-ended => returns (None, False)
    - On invalid string => returns (None, True) and logs a clear error
    """
    if value is None:
        return None, False
    if isinstance(value, str) and value.strip() == "":
        return None, False

    try:
        ts = pd.to_datetime(value, errors="raise")
        # ensure tz-naive to match typical feather storage
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(None) if hasattr(ts, "tz_convert") else ts.tz_localize(None)
        return pd.Timestamp(ts), False
    except Exception:
        console = Console()

        console.print(
            f"[bold red]⛔ [DATE] Invalid {field_name}: '{value}'[/bold red]\n"
            "[yellow]👉 Use a valid datetime like '2025-08-01 08:31:00' (YYYY-MM-DD HH:MM:SS)[/yellow]"
        )
        return None, True

def load_data(file_path, start_date=None, end_date=None, resample_freq=None):
    """
    Load OHLCV data from Feather, filter by [start_date, end_date],
    rename columns, set timestamp index, and optionally resample.

    On invalid start_date/end_date:
      - Log a clear error
      - Return an EMPTY DataFrame (no exception)
    """
    # ★ always target .feather
    feather_path = file_path.replace('.csv', '.feather')
    if not os.path.exists(feather_path):
        # [CHANGED] return empty with error instead of raising (keeps your "no exception" rule consistent)
        logger.error("Feather file not found: %s", feather_path)
        return _empty_ohlcv_like()

    logger.info(f"Loading data from {feather_path}...")

    # ★ direct Feather load
    df = pd.read_feather(feather_path)

    # [ADDED] ensure timestamp column is datetime64[ns]
    if "timestamp" not in df.columns:
        logger.error("Missing 'timestamp' column in %s.", feather_path)
        return _empty_ohlcv_like()
    # coerce bad timestamps to NaT, then drop them (safer than raising later)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    bad_ts = df["timestamp"].isna().sum()
    if bad_ts:
        warnings.warn(f"{bad_ts} rows with unparsable timestamps were dropped.")
        df = df.dropna(subset=["timestamp"])

    # [ADDED] parse dates safely
    start_ts, start_invalid = _safe_parse_datetime(start_date, "start_date")
    end_ts, end_invalid     = _safe_parse_datetime(end_date,   "end_date")

    # [ADDED] if either date is invalid -> return empty (no exception)
    if start_invalid or end_invalid:
        return _empty_ohlcv_like()

    # [ADDED] logical ordering check: start <= end
    if start_ts is not None and end_ts is not None and start_ts > end_ts:
        logger.error(
            "[DATE] start_date (%s) must be earlier than or equal to end_date (%s).",
            start_ts, end_ts
        )
        return _empty_ohlcv_like()

    # ★ filter BEFORE indexing (using parsed timestamps)
    if start_ts is not None:
        df = df[df['timestamp'] >= start_ts]
    if end_ts is not None:
        df = df[df['timestamp'] <= end_ts]

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
        # [CHANGED] log and return empty instead of raising
        logger.error("Missing required columns: %s", missing_cols)
        return _empty_ohlcv_like(sorted(required_columns))

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
        # resampling empty frames returns empty as well, which is fine
        df = df.resample(resample_freq).last().ffill()

    # Print debugging information about the final data
    if not df.empty:
        logger.info(f"Final data range: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Final data count: {len(df)} rows")
        if resample_freq:
            logger.info(f"Resample frequency: {resample_freq}")
    else:
        logger.warning("Final data is empty after processing")

    # Done
    return df
