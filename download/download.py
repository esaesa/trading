#download.py
import json
from pathlib import Path
import time
import ccxt
import pytz
import pandas as pd
import os
from datetime import datetime, timezone
from common_utils import beep, load_config
from tqdm import tqdm  # New: progress bar package
import math
from logger_config import logger  # Centralized logger
# --- Supported Binance timeframes ---
SUPPORTED_TIMEFRAMES = {
    '1m', '3m', '5m', '15m', '30m',
    '1h', '2h', '4h', '6h', '8h', '12h',
    '1d', '3d', '1w', '1M'
}
def validate_timeframe(timeframe):
    """
    Validate that the timeframe is supported by Binance.
    Raises ValueError if not supported.
    """
    if timeframe not in SUPPORTED_TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe}. Supported: {SUPPORTED_TIMEFRAMES}")
    
# --- Function to resolve the data path ---
def resolve_data_path(config):
    """
    Resolve the data path from the configuration.
    Supports both relative and absolute paths.
    """
    data_path = config.get('data_path', 'data')  # Default to './data' if not specified
    if not os.path.isabs(data_path):  # Check if the path is relative
        script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
        data_path = os.path.join(script_dir, data_path)  # Convert to absolute path
    return data_path

def safe_fetch_ohlcv( symbol, timeframe, since_ms, limit):
    time.sleep(0.1)  # Conservative rate limiting
    return exchange.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=limit)

# NEW: Build absolute path to the cache file
script_dir = os.path.dirname(os.path.abspath(__file__))  + "/data"
cache_path = os.path.join(script_dir, "binance_oldest_timestamp_cache.json")

def load_cache():
    """Load the cache from a JSON file in this script's folder."""
    if Path(cache_path).exists():
        with open(cache_path, "r") as f:
            cache = json.load(f)
        # Convert cached timestamps (ints) back to datetime (UTC)
        return {k: datetime.fromtimestamp(v / 1000, timezone.utc) for k, v in cache.items() if isinstance(v, int)}
    return {}

def save_cache(cache):
    """Save the cache to a JSON file in this script's folder."""
    # Convert datetime objects to timestamps (ms)
    sanitized_cache = {
        k: int(v.timestamp() * 1000) if isinstance(v, datetime) else v 
        for k, v in cache.items()
    }
    with open(cache_path, "w") as f:
        json.dump(sanitized_cache, f)

def round_down_time(time_ms, timeframe_ms):
    return (time_ms // timeframe_ms) * timeframe_ms

logger.info("Starting data download...")

# --- Helper function for timezone conversion ---
def convert_to_utc(local_time_str, tz_str="UTC"):
    if tz_str not in pytz.all_timezones:
        raise ValueError(f"Invalid timezone: {tz_str}. Use a valid timezone from pytz.all_timezones.")
    tz = pytz.timezone(tz_str)
    naive_dt = datetime.strptime(local_time_str, '%Y-%m-%d %H:%M:%S')
    aware_dt = tz.localize(naive_dt, is_dst=None)  # Raises exception if ambiguous
    return aware_dt.astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
# def convert_to_utc(local_time_str, tz_str="UTC", fmt='%Y-%m-%d %H:%M:%S'):
#     """
#     Convert a local time string in timezone tz_str to a UTC time string.
#     If the provided string does not include time information, it defaults to midnight.
#     """
#     # If the string length suggests no time info, append " 00:00:00"
#     if len(local_time_str.strip()) == 10:
#         local_time_str = local_time_str.strip() + " 00:00:00"
#     local_tz = pytz.timezone(tz_str)
#     local_dt = datetime.strptime(local_time_str, fmt)
#     local_dt = local_tz.localize(local_dt)
#     utc_dt = local_dt.astimezone(pytz.utc)
#     return utc_dt.strftime(fmt)


# --- Helper: convert timeframe string to milliseconds ---
def timeframe_to_milliseconds(timeframe):
    """
    Convert timeframe string (e.g., '1m', '5m', '1h', '1d') to milliseconds.
    """
    unit = timeframe[-1]
    amount = int(timeframe[:-1])
    if unit == 'm':
        return amount * 60 * 1000
    elif unit == 'h':
        return amount * 60 * 60 * 1000
    elif unit == 'd':
        return amount * 24 * 60 * 60 * 1000
    else:
        raise ValueError("Unsupported timeframe unit")

# --- Function to get the oldest available Binance timestamp ---
def get_binance_oldest_timestamp(exchange, symbol, timeframe):
    """Find the true oldest available timestamp by incrementally querying Binance."""
    # Load the cache
    cache = load_cache()
    cache_key = f"{symbol}_{timeframe}"
    
    # Check if the timestamp is already cached
    if cache_key in cache:
        logger.info(f"📌 Found cached oldest timestamp for {symbol} [{timeframe}]: {cache[cache_key]}")
        return cache[cache_key]  

   
    start_year = 2017  # Binance started in 2017
    step_months = 6  # Step in 6-month intervals until data is found

    for year in range(start_year, datetime.now(timezone.utc).year + 1):
        for month in range(1, 13, step_months):
            test_date = datetime(year, month, 1)
            test_timestamp_ms = int(test_date.timestamp() * 1000)
            try:
                ohlcv =safe_fetch_ohlcv(symbol, timeframe, since=test_timestamp_ms, limit=1)
                if ohlcv:
                    oldest_time = datetime.fromtimestamp(ohlcv[0][0] / 1000, timezone.utc)
                    logger.info(f"📌 Found first available Binance data: {oldest_time}")
                    # Save to cache
                    cache[cache_key] = oldest_time
                    save_cache(cache)
                    return oldest_time
            except Exception as e:
                logger.warning(f"⚠️ Error while searching for oldest timestamp at {test_date}: {e}")
    logger.warning("⚠️ Unable to determine oldest available data.")
    return None

# --- Function to fetch OHLCV data with an optional end time ---
def fetch_new_binance_ohlcv(exchange, symbol, timeframe, since, end=None):
    """
    Fetch new OHLCV data from Binance from the given 'since' time up to an optional 'end' time.
    Times must be provided as UTC strings in the format '%Y-%m-%d %H:%M:%S'.
    """
    if len(since) == 10:
        since += " 00:00:00"
    logger.info(f"🔍 Fetching data from {since} to {end} for {symbol} [{timeframe}]...")

    since_dt = datetime.strptime(since, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    since_ms = int(since_dt.timestamp() * 1000)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if end:
        if len(end) == 10:
             # Parse the date string and add one day minus one second
            end += " 23:59:59"
        end_dt = datetime.strptime(end, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)   
        end_ms = int(end_dt.timestamp() * 1000)
        effective_end_ms = min(now_ms, end_ms)
    else:
        effective_end_ms = now_ms

    all_data = []
    limit = 1000

    # --- Setup progress bar ---
    period_ms = timeframe_to_milliseconds(timeframe)
    total_expected_candles = math.ceil((effective_end_ms - since_ms) / period_ms)
    pbar = tqdm(total=total_expected_candles, desc="Downloading candles", unit="candle")

    while since_ms < effective_end_ms:
        try:
            # Fetch OHLCV data in batches of 1000 candles
            #logger.debug(f"Fetching {symbol} [{timeframe}] from {since_ms} limit {limit}...")
            ohlcv = safe_fetch_ohlcv(symbol=symbol, timeframe=timeframe, since_ms=since_ms, limit=limit)
        except Exception as e:
            logger.error(f"⚠️ Error fetching data at timestamp {since_ms}: {e}", exc_info=True)
            break

        if not ohlcv:
            break

        if end:
            # Filter out candles with timestamps >= effective_end_ms
            ohlcv = [row for row in ohlcv if row[0] < effective_end_ms]
            if not ohlcv:
                break

        if not all(ohlcv[i][0] < ohlcv[i+1][0] for i in range(len(ohlcv)-1)):
            logger.warning("⚠️ Fetched OHLCV data is not sorted! Sorting now.")
            ohlcv.sort(key=lambda x: x[0])  # Sort by timestamp
        all_data.extend(ohlcv)
        # Update progress bar by number of candles downloaded in this batch.
        pbar.update(len(ohlcv))

        since_ms = ohlcv[-1][0] + 1  # Move forward to avoid duplicates

       

    pbar.close()
    if not all(all_data[i][0] < all_data[i+1][0] for i in range(len(all_data)-1)):
        logger.warning("⚠️ Combined OHLCV data is not sorted! Sorting now.")
        all_data.sort(key=lambda x: x[0])  # Sort by timestamp
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    if not df.empty:
        start_time = df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
        end_time = df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"✅ Downloaded {len(df)} new rows. (From {start_time} to {end_time})")
    else:
        logger.warning("⚠️ No new data downloaded.")

    return df

# --- Main execution ---
if __name__ == "__main__":
    # Load configuration from file
    config_dir = os.path.dirname(__file__)
    config_path = os.path.join(config_dir, "config.json")
    config = load_config(config_path)
    download_config = config.get('download', {})

    # Extract configuration details
    api_key = download_config['apiKey']
    api_secret = download_config['apiSecret']
    exchange_name = download_config['exchange']
    symbols = download_config.get('symbols', [])  # List of symbols
    timeframes = download_config.get('timeframes', [])  # List of timeframes
    local_start = download_config['start_date']  # Start date in local timezone
    local_timezone = download_config.get('timezone', 'UTC')  # Default to UTC if not specified
    
    # Validate timeframes
    for timeframe in timeframes:
        try:
            validate_timeframe(timeframe)
        except ValueError as e:
            logger.error(f"❌ Invalid timeframe configuration: {e}")
            logger.info("🛑 Aborting script due to invalid timeframe.")
            exit(1)

    # Handle optional end_date
    if 'end_date' in download_config:
        local_end = download_config['end_date']
        utc_end_str = convert_to_utc(local_end, local_timezone)
        logger.info(f"Converted local end {local_end} ({local_timezone}) to UTC end: {utc_end_str}")
    else:
        utc_end_str = None  # No end date specified

    # Create the exchange instance
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True
    })

    # Resolve the data path
    data_folder = resolve_data_path(download_config)
    os.makedirs(data_folder, exist_ok=True)  # Ensure the folder exists

    # Convert the local start date to UTC
    utc_start_str = convert_to_utc(local_start, local_timezone)
    logger.info(f"Converted local startt {local_start} ({local_timezone}) to UTC start: {utc_start_str}")



    # Loop through each symbol and timeframe combination
    for symbol in symbols:
        for timeframe in timeframes:
            logger.info(f"🚀 Processing {symbol} with timeframe {timeframe}...")

            # Set the file name for the consolidated dataset
            safe_symbol = symbol.replace("/", "").replace(":", "")  # Remove invalid characters
            full_data_filename = os.path.join(data_folder, f"{exchange_name}_{safe_symbol}_{timeframe}.csv")

            # Get the oldest available timestamp on Binance for this symbol and timeframe
            oldest_binance_timestamp = get_binance_oldest_timestamp(exchange, symbol, timeframe)

            if os.path.exists(full_data_filename):
                logger.info(f"📂 Found existing dataset: {full_data_filename}")
                df_existing = pd.read_csv(full_data_filename)
                df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'])

                # Get the earliest and latest timestamps from the existing data
                earliest_existing_timestamp = df_existing['timestamp'].min().replace(tzinfo=timezone.utc)
                latest_existing_timestamp = df_existing['timestamp'].max()
                logger.info(f"🕒 Existing data range: {earliest_existing_timestamp} to {latest_existing_timestamp}")

                # Parse the UTC start date string into a datetime object
                config_start_date = datetime.strptime(utc_start_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)

                # Adjust start date if it is earlier than the oldest available Binance data
                if oldest_binance_timestamp and config_start_date < oldest_binance_timestamp:
                    logger.warning(f"⚠️ Requested start date {config_start_date} is too old! Adjusting to {oldest_binance_timestamp}.")
                    config_start_date = oldest_binance_timestamp

                # Fetch older data if the config start (UTC) is earlier than what you already have
                if config_start_date < earliest_existing_timestamp:
                    logger.info(f"⬇️ Need to fetch older data from {config_start_date.strftime('%Y-%m-%d %H:%M:%S')} to {earliest_existing_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                    #fetch older data till the earliest existing timestamp
                    df_old = fetch_new_binance_ohlcv(exchange, symbol, timeframe, since=config_start_date.strftime('%Y-%m-%d %H:%M:%S'), end=earliest_existing_timestamp.strftime('%Y-%m-%d %H:%M:%S'))
                    #exit()
                    #clean existing data in case of duplicates came from exchange
                    df_old = df_old[~df_old['timestamp'].isin(df_existing['timestamp'])]
                else:
                    df_old = pd.DataFrame()  # No older data to fetch

                
                # Fetch newer data starting from the latest existing timestamp till the end date
                #since_date = latest_existing_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                # Add 1 millisecond to avoid fetching the same candle
                since_date = (latest_existing_timestamp + pd.Timedelta(milliseconds=1)).strftime('%Y-%m-%d %H:%M:%S')
                #logger.info(f"🔄 Latest existing timestamp: {since_date}, UTC end: {utc_end_str}")
                logger.info(f"🔄 Latest existing timestamp: {latest_existing_timestamp}, New since: {since_date}")
                if utc_end_str and datetime.strptime(utc_end_str, '%Y-%m-%d %H:%M:%S') <= latest_existing_timestamp:
                    logger.info(f"✅ Existing data already covers the required end date ({utc_end_str}). No new data to fetch.")
                    df_new = pd.DataFrame()  # No newer data to fetch
                else:
                    logger.info(f"🔄 Fetching new data from {since_date} to {utc_end_str}")
                    df_new = fetch_new_binance_ohlcv(exchange, symbol, timeframe,since= since_date,end=utc_end_str)
                    df_new = df_new[~df_new['timestamp'].isin(df_existing['timestamp'])]

                # Merge the old, existing, and new data
                # Filter out empty DataFrames before concatenation
                # Check if both df_old and df_new are empty
                if df_old.empty and df_new.empty:
                    logger.info("✅ Dataset is up-to-date. No changes made.")
                else:
                    dataframes_to_concat = [df for df in [df_old, df_existing, df_new] if not df.empty]
                    logger.info(f"🔄 Merging data: df_old={len(df_old)}, df_existing={len(df_existing)}, df_new={len(df_new)}")

                    if dataframes_to_concat:  # Check if there are any non-empty DataFrames
                        df_combined = pd.concat(dataframes_to_concat).reset_index(drop=True)
                    else:
                        df_combined = pd.DataFrame()  # Handle the case where all DataFrames are empty

                    if not df_combined['timestamp'].is_monotonic_increasing:
                        logger.warning("⚠️ DataFrame is not sorted by 'timestamp'! Sorting now.")
                        df_combined.sort_values(by='timestamp', inplace=True)

                    logger.info(f"📊 Final dataset size after merge: {len(df_combined)} rows")
                    
                    # Check for gaps in the OHLCV data
                    timeframe_ms = timeframe_to_milliseconds(timeframe)
                    gaps = df_combined['timestamp'].diff().dt.total_seconds() * 1000 > 2 * timeframe_ms
                    if gaps.any():
                        logger.warning("⚠️ Gaps detected in OHLCV data. Check for missing or duplicated candles.")

                    # Save the combined dataset to CSV
                    if df_combined.empty:
                        logger.warning("⚠️ Combined dataset is empty. Skipping save.")
                    else:
                        assert all(col in df_combined.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']), "Missing required OHLCV columns"
                        df_combined.to_csv(full_data_filename, index=False)
                        logger.info(f"✅ Full dataset updated: {full_data_filename}")



            else:
                # No existing file: fetch full data starting from the config start date (UTC)
                logger.info("🆕 No existing dataset found. Downloading full dataset...")
                config_start_date = datetime.strptime(utc_start_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)

                if oldest_binance_timestamp and config_start_date < oldest_binance_timestamp:
                    logger.warning(f"⚠️ Requested start date {config_start_date} is too old! Adjusting to {oldest_binance_timestamp}.")
                    config_start_date = oldest_binance_timestamp

                df_new = fetch_new_binance_ohlcv(exchange, symbol, timeframe, since=config_start_date.strftime('%Y-%m-%d %H:%M:%S'), end=utc_end_str)

                if not df_new.empty:
                    df_new.to_csv(full_data_filename, index=False)
                    logger.info(f"✅ Full dataset saved to {full_data_filename}")
                else:
                    logger.warning("⚠️ No data fetched.")

    logger.info("✨ All data download tasks completed.")
    beep()