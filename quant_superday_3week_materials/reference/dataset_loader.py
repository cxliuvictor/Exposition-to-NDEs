
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def load_checked_csv(name, url, expected_cols=None, min_rows=None, read_csv_kwargs=None, force=False):
    read_csv_kwargs = read_csv_kwargs or {}
    cache_path = CACHE_DIR / name
    if cache_path.exists() and not force:
        df = pd.read_csv(cache_path)
    else:
        df = pd.read_csv(url, **read_csv_kwargs)
        df.to_csv(cache_path, index=False)
    if expected_cols:
        missing = set(expected_cols) - set(df.columns)
        assert not missing, "Missing expected columns: {}".format(sorted(missing))
    if min_rows is not None:
        assert len(df) >= min_rows, "Dataset unexpectedly short: {} rows".format(len(df))
    return df
