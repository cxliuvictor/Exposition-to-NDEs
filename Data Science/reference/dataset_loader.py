from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
CACHE=ROOT/'data'/'cache'
CACHE.mkdir(parents=True, exist_ok=True)
def load_checked_csv(name,url,expected_cols=None,min_rows=None,read_csv_kwargs=None,force=False):
    read_csv_kwargs=read_csv_kwargs or {}
    p=CACHE/name
    if p.exists() and not force:
        df=pd.read_csv(p)
    else:
        df=pd.read_csv(url,**read_csv_kwargs)
        df.to_csv(p,index=False)
    if expected_cols:
        missing=set(expected_cols)-set(df.columns)
        assert not missing, f'Missing expected columns: {sorted(missing)}'
    if min_rows is not None:
        assert len(df)>=min_rows, f'Unexpectedly short dataset: {len(df)} rows'
    return df
