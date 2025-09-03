# src/transforms/pruning.py
import pandas as pd

def drop_sparse_columns(df: pd.DataFrame, min_non_null_frac: float = 0.20) -> pd.DataFrame:
    if df is None or df.empty: return df
    thresh = int(min_non_null_frac * len(df))
    return df.dropna(axis=1, thresh=thresh)
