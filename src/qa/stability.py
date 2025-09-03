# src/diagnostics/stability.py
import numpy as np
import pandas as pd
from typing import Sequence, Dict, Optional, Tuple

def ic_time_series(df: pd.DataFrame, feature_cols: Sequence[str], ret_col: str,
                   window: int = 60, method: str = "pearson") -> pd.DataFrame:
    """
    Rolling Information Coefficient per feature vs a continuous target (e.g., fwd return).
    Returns wide DF with one column per feature (IC_t).
    """
    assert method in ("pearson","spearman")
    y = pd.to_numeric(df[ret_col], errors="coerce")
    out = {}
    for f in feature_cols:
        x = pd.to_numeric(df[f], errors="coerce")
        if method == "pearson":
            roll = x.rolling(window).corr(y)
        else:
            roll = x.rolling(window).apply(lambda s: pd.Series(s).rank().corr(pd.Series(y.loc[s.index]).rank()), raw=False)
        out[f"{f}__IC_{method}_{window}"] = roll
    return pd.DataFrame(out, index=df.index)

def topk_turnover(feature_scores_by_date: pd.DataFrame, k: int = 20) -> pd.Series:
    """
    feature_scores_by_date: rows = dates, cols = features, values = score (e.g., IC at date)
    Turnover at t is 1 - |A_t ∩ A_{t-1}| / |A_{t-1}| for top-k sets.
    """
    idx = feature_scores_by_date.index
    turnover = pd.Series(index=idx, dtype=float)
    prev = None
    for t in idx:
        row = feature_scores_by_date.loc[t].dropna()
        if row.empty:
            turnover.loc[t] = np.nan
            continue
        top = set(row.sort_values(ascending=False).head(k).index)
        if prev is None:
            turnover.loc[t] = np.nan
        else:
            turnover.loc[t] = 1.0 - (len(top & prev) / (len(prev) + 1e-9))
        prev = top
    return turnover.rename("topk_turnover")

def summarize_ic_stability(ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summaries: median IC, IQR, %positive, |IR| estimate.
    """
    med = ic_df.median(skipna=True)
    iqr = (ic_df.quantile(0.75) - ic_df.quantile(0.25))
    pos = (ic_df > 0).mean()
    ann_ir = med / (ic_df.std(skipna=True) + 1e-12) * np.sqrt(252)  # rough
    return pd.DataFrame({"medianIC": med, "IQR": iqr, "frac_pos": pos, "ann_IR_est": ann_ir})

def shap_cv_stability(models: Sequence, X_folds: Sequence[pd.DataFrame], explainer=None) -> pd.DataFrame:
    """
    Optional: compute mean(|corr|) of SHAP values across folds for common features.
    Requires tree models (or pass a compatible explainer).
    """
    try:
        import shap  # only if installed
    except Exception:
        # return empty if not available
        return pd.DataFrame()

    if explainer is None:
        explainer = shap.TreeExplainer(models[0])
    shap_values = []
    for m, X in zip(models, X_folds):
        e = shap.TreeExplainer(m)
        sv = e.shap_values(X, check_additivity=False)
        # for binary clf, pick sv[1] if needed; otherwise coerce to array
        arr = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv
        shap_values.append(pd.DataFrame(arr, columns=X.columns, index=X.index))

    # compute pairwise correlations of per-feature SHAP across folds then average
    cols = list(set.intersection(*[set(sv.columns) for sv in shap_values]))
    if not cols:
        return pd.DataFrame()
    corr_mat = {}
    for c in cols:
        # stack each fold’s SHAP series aligned by index name (ignore exact date overlap)
        series = [sv[c].reset_index(drop=True) for sv in shap_values]
        # mean of pairwise |corr|
        vals = []
        for i in range(len(series)):
            for j in range(i+1, len(series)):
                r = np.corrcoef(series[i].fillna(0), series[j].fillna(0))[0,1]
                vals.append(abs(r))
        corr_mat[c] = np.mean(vals) if vals else np.nan
    return pd.DataFrame.from_dict(corr_mat, orient="index", columns=["mean_abs_corr_across_folds"]).sort_values(“mean_abs_corr_across_folds”, ascending=False)
