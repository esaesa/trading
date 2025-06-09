import pandas as pd
import numpy as np
import time
import os

def generate_data(n_rows: int) -> pd.DataFrame:
    """Generate n_rows of datetime, price, high, low, volume."""
    dt = pd.date_range("2025-01-01", periods=n_rows, freq="S")
    price = np.cumsum(np.random.randn(n_rows)) + 10000  # random walk around 10k
    high = price + np.random.rand(n_rows) * 10
    low = price - np.random.rand(n_rows) * 10
    volume = np.random.rand(n_rows) * 5
    return pd.DataFrame({
        "datetime": dt,
        "price": price,
        "high": high,
        "low": low,
        "volume": volume
    })

def time_op(func, *args, **kwargs):
    """Time func(*args, **kwargs), return (result, elapsed_sec)."""
    start = time.perf_counter()
    res = func(*args, **kwargs)
    return res, time.perf_counter() - start

def compare_formats(df: pd.DataFrame, base_path: str):
    csv_path = base_path + ".csv"
    feather_path = base_path + ".feather"
    results = {}

    # WRITE CSV
    _, t_csv_w = time_op(df.to_csv, csv_path, index=False)
    results["csv_write_time"] = t_csv_w
    results["csv_write_tp"] = len(df) / t_csv_w

    # WRITE Feather
    _, t_fth_w = time_op(df.to_feather, feather_path)
    results["feather_write_time"] = t_fth_w
    results["feather_write_tp"] = len(df) / t_fth_w

    # READ CSV
    _, t_csv_r = time_op(pd.read_csv, csv_path, parse_dates=["datetime"])
    results["csv_read_time"] = t_csv_r
    results["csv_read_tp"] = len(df) / t_csv_r

    # READ Feather
    _, t_fth_r = time_op(pd.read_feather, feather_path)
    results["feather_read_time"] = t_fth_r
    results["feather_read_tp"] = len(df) / t_fth_r

    # Clean up
    os.remove(csv_path)
    os.remove(feather_path)

    return results

def run_example(n_rows, idx):
    print(f"\nExample {idx}: {n_rows:,} rows")
    df = generate_data(n_rows)
    res = compare_formats(df, f"crypto_data_{n_rows}")
    print(f"  CSV write:    {res['csv_write_time']:.3f}s  ({res['csv_write_tp']:.0f} rows/s)")
    print(f"  Feather write:{res['feather_write_time']:.3f}s  ({res['feather_write_tp']:.0f} rows/s)")
    print(f"  CSV read:     {res['csv_read_time']:.3f}s  ({res['csv_read_tp']:.0f} rows/s)")
    print(f"  Feather read: {res['feather_read_time']:.3f}s  ({res['feather_read_tp']:.0f} rows/s)")

if __name__ == "__main__":
    # Numbered examples with increasing size to show linear scaling:
    run_example(100_000, idx=1)
    run_example(1_000_000, idx=2)
