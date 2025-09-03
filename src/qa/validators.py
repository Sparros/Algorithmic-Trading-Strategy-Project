import numpy as np
import pandas as pd

def validate_df(df: pd.DataFrame, name: str = "", require_dtindex: bool = True) -> pd.DataFrame:
    """
    - Coerce datetime index (if requested)
    - Drop tz info
    - Drop duplicate index rows
    - Replace inf with NaN
    - Drop all-NaN columns
    - Sort index
    """
    if df is None or len(df) == 0:
        return df

    if require_dtindex and not isinstance(df.index, pd.DatetimeIndex):
        try:
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], utc=False, errors='coerce')
                df = df.set_index('Date')
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], utc=False, errors='coerce')
                df = df.set_index('date')
            else:
                df.index = pd.to_datetime(df.index, utc=False, errors='coerce')
        except Exception as e:
            print(f"[validate] {name}: failed to coerce datetime index ({e})")

    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    before = len(df)
    df = df[~df.index.duplicated(keep='first')].sort_index()
    if len(df) != before:
        print(f"[validate] {name}: removed {before - len(df)} duplicate index rows")

    numeric = df.select_dtypes(include=[np.number])
    if not numeric.empty:
        mask = ~np.isfinite(numeric)
        if mask.values.any():
            n_infs = int(mask.values.sum())
            df = df.replace([np.inf, -np.inf], np.nan)
            print(f"[validate] {name}: replaced {n_infs} ±inf with NaN")

    all_na_cols = [c for c in df.columns if df[c].isna().all()]
    if all_na_cols:
        print(f"[validate] {name}: dropping all-NaN columns: {all_na_cols}")
        df = df.drop(columns=all_na_cols)

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    return df

def assert_monotonic(index: pd.Index):
    if not index.is_monotonic_increasing:
        raise AssertionError("Index not monotonic increasing")

def assert_no_future(stock_idx: pd.Index, macro_idx: pd.Index, tolerance_days: int = 31):
    """
    Guarantee macro availability does not peek into the future relative to stocks.
    We allow macro to lag — but if macro ends far behind last stock date, raise.
    """
    if len(stock_idx) == 0 or len(macro_idx) == 0:
        return
    if stock_idx.max() > macro_idx.max():
        lag_days = (stock_idx.max() - macro_idx.max()).days
        if lag_days > tolerance_days:
            raise AssertionError(f"Macro data ends {lag_days} days before last stock date")

def quick_index_stats(idx: pd.DatetimeIndex) -> dict:
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) == 0:
        return dict(rows=0, dup_index_rows=0, monotonic=False, tz_naive=True, median_gap_days=np.nan)
    gaps = pd.Series(idx).diff().dropna()
    med_gap_days = float(gaps.median().days) if not gaps.empty else np.nan
    return dict(
        rows=int(len(idx)),
        dup_index_rows=int(pd.Index(idx).duplicated().sum()),
        monotonic=bool(idx.is_monotonic_increasing),
        tz_naive=(getattr(idx, "tz", None) is None),
        median_gap_days=med_gap_days
    )
