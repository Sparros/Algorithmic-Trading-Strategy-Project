import json
import numpy as np
import pandas as pd

def _year_series(idx_or_df) -> pd.Series:
    if isinstance(idx_or_df, pd.DatetimeIndex):
        return pd.Series(idx_or_df.year, index=idx_or_df)
    if isinstance(idx_or_df, pd.Series) and isinstance(idx_or_df.index, pd.DatetimeIndex):
        return pd.Series(idx_or_df.index.year, index=idx_or_df.index)
    if isinstance(idx_or_df, pd.DataFrame) and "Date" in idx_or_df:
        return pd.to_datetime(idx_or_df["Date"]).dt.year
    return pd.Series([np.nan]*len(idx_or_df))

def _run_lengths(binary_series: pd.Series):
    """
    Return run-length list for 1's (and for 0's via 1-x if needed).
    """
    if binary_series.empty:
        return []
    x = binary_series.astype(int)
    # group on changes
    g = (x != x.shift()).cumsum()
    # lengths of each run
    lens = x.groupby(g).cumcount() + 1
    # capture only the ends
    ends = (x.diff().fillna(0) != 0)
    return lens[ends].tolist()

def target_health_report(df: pd.DataFrame, ticker: str,
                         windows=(1,5,10), thresholds=(0.0, 0.0025, 0.005),
                         min_samples: int = 150) -> pd.DataFrame:
    """
    Ephemeral target diagnostics for QA. Does NOT write back to disk.
    Returns wide table with coverage, base rates, drift, and run-length stats.
    """
    rows = []
    close_col = f"Close_{ticker}"
    if close_col not in df.columns:
        return pd.DataFrame([dict(
            window=np.nan, threshold=np.nan, eff_samples=0, pos_rate=np.nan,
            pos_count=0, neg_count=0, pos_rate_by_year={},
            median_ret=np.nan, q10=np.nan, q90=np.nan,
            mean_run_up=np.nan, mean_run_down=np.nan, issues=f"missing_close_col({close_col})"
        )])

    for w in windows:
        fwd = df[close_col].pct_change(periods=w).shift(-w)
        eff_samples = int(fwd.notna().sum())
        if eff_samples < min_samples:
            rows.append(dict(
                window=w, threshold=np.nan, eff_samples=eff_samples, pos_rate=np.nan,
                pos_count=0, neg_count=0, pos_rate_by_year={},
                median_ret=np.nan, q10=np.nan, q90=np.nan,
                mean_run_up=np.nan, mean_run_down=np.nan, issues="insufficient_samples"
            ))
            continue

        med = float(fwd.median())
        q10 = float(fwd.quantile(0.10))
        q90 = float(fwd.quantile(0.90))

        # run-lengths at 0-threshold
        sign0 = (fwd > 0).astype(int)
        up_runs = _run_lengths(sign0.replace(0, np.nan).dropna())
        dn_runs = _run_lengths((1 - sign0).replace(0, np.nan).dropna())

        for t in thresholds:
            y = (fwd > t).astype(float)
            pos_rate = float(y.mean())
            pos_count = int(y.sum())
            neg_count = int((1 - y).sum())

            yrs = _year_series(df.index if isinstance(df.index, pd.DatetimeIndex) else df["Date"])
            yr_tbl = (pd.DataFrame({"y": y, "year": yrs})
                      .dropna()
                      .groupby("year")["y"].mean()
                      .round(3)
                      .to_dict())

            # flags
            issues = []
            if pos_rate < 0.05 or pos_rate > 0.95:
                issues.append("extreme_imbalance")
            if len(yr_tbl) >= 2:
                rng = max(yr_tbl.values()) - min(yr_tbl.values())
                if rng > 0.25:
                    issues.append("base_rate_drift")

            rows.append(dict(
                window=w, threshold=float(t),
                eff_samples=eff_samples,
                pos_rate=round(pos_rate, 3),
                pos_count=pos_count, neg_count=neg_count,
                pos_rate_by_year=yr_tbl,
                median_ret=round(med, 6),
                q10=round(q10, 6), q90=round(q90, 6),
                mean_run_up=float(np.mean(up_runs)) if up_runs else np.nan,
                mean_run_down=float(np.mean(dn_runs)) if dn_runs else np.nan,
                issues=",".join(issues)
            ))
    return pd.DataFrame(rows)

def suggest_thresholds_from_distribution(df: pd.DataFrame, ticker: str,
                                         window: int, target_pos_rate: float = 0.5) -> dict:
    """
    Suggest threshold t so that P(fwd_ret > t) ≈ target_pos_rate.
    """
    close_col = f"Close_{ticker}"
    if close_col not in df.columns:
        return {"window": window, "suggested_threshold": None, "note": f"missing {close_col}"}
    fwd = df[close_col].pct_change(periods=window).shift(-window).dropna()
    if fwd.empty:
        return {"window": window, "suggested_threshold": None, "note": "no data"}
    t = float(np.quantile(fwd, 1 - target_pos_rate))
    return {"window": window, "suggested_threshold": round(t, 6), "note": ""}
