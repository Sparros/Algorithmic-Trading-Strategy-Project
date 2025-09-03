import numpy as np
import pandas as pd

def percent_nan(df: pd.DataFrame) -> float:
    return float(df.isna().mean().mean()) if len(df) else 0.0

def days_since_change(series: pd.Series) -> pd.Series:
    """
    Approximate staleness in rows since last change.
    Assumes upstream business-day spacing.
    """
    changed = series.ne(series.shift(1)).astype(int)
    grp = changed.cumsum()
    # position within current streak (0-based)
    pos = series.groupby(grp).cumcount()
    return pos

def find_identical_cols(df: pd.DataFrame, sample: int = 5000):
    """
    Return pairs of columns that are exactly identical on a sample (or full set if small).
    """
    cols = df.columns.tolist()
    pairs = []
    sub = df.sample(min(len(df), sample), random_state=42) if len(df) > sample else df
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            a, b = cols[i], cols[j]
            if sub[a].equals(sub[b]):
                pairs.append((a, b))
    return pairs

def qa_one_file(path: str) -> dict:
    import os
    name = os.path.basename(path)
    try:
        df = pd.read_csv(path, parse_dates=["Date"])
        df = df.sort_values("Date").set_index("Date")
    except Exception as e:
        return {"dataset": name, "ok": False, "error": f"read_error: {e}"}

    idx = df.index
    monotonic = bool(idx.is_monotonic_increasing)
    tz_ok = getattr(idx, "tz", None) is None
    dup_idx = int(pd.Index(idx).duplicated().sum())

    # calendar stats
    gaps = pd.Series(idx).diff().dropna()
    med_gap_days = float(gaps.median().days) if not gaps.empty else np.nan

    # missingness
    nan_overall = percent_nan(df)
    sparse_thresh = 0.80
    n_sparse = int((df.isna().mean() > sparse_thresh).sum())

    # near-constant features
    num = df.select_dtypes(include=[np.number])
    variances = num.var(ddof=0) if not num.empty else pd.Series(dtype=float)
    near_const = variances[variances < 1e-10].index.tolist()

    # macro staleness (approx)
    macro_cols = [c for c in df.columns if c.isupper() and len(c) >= 4 and not c.startswith(("Open_","High_","Low_","Close_","Volume_"))]
    macro_cols = [c for c in macro_cols if c in num.columns]
    max_stale = {}
    for m in macro_cols[:50]:
        st = days_since_change(df[m])
        max_stale[m] = int(st.max()) if len(st) else 0
    worst_macro_staleness = max(max_stale.values()) if max_stale else 0

    # identical columns (sample-based)
    dup_feature_pairs = find_identical_cols(num) if num.shape[1] <= 1000 else []
    n_identical_pairs = len(dup_feature_pairs)

    return dict(
        dataset=name, ok=True, error="",
        rows=int(len(df)), cols=int(df.shape[1]),
        idx_monotonic=monotonic, tz_naive=tz_ok, dup_index_rows=dup_idx,
        median_gap_days=med_gap_days,
        nan_overall=round(nan_overall, 4),
        n_sparse_cols=n_sparse,
        n_near_constant=len(near_const),
        worst_macro_staleness_days=int(worst_macro_staleness),
        n_identical_pairs=n_identical_pairs
    )
