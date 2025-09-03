# src/qa/leakage.py
import numpy as np
import pandas as pd
from typing import Callable, Dict, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, r2_score

def permutation_target_test(
    X: pd.DataFrame,
    y: pd.Series,
    fit_fn: Callable[[pd.DataFrame, pd.Series], Any],
    pred_fn: Callable[[Any, pd.DataFrame], np.ndarray],
    n_splits: int = 5,
    n_permutations: int = 50,
    metric: str = "roc_auc"
) -> Dict[str, float]:
    """
    Train/evaluate with true labels, then with shuffled labels.
    If model retains high performance on shuffled labels, suspect leakage.
    metric: 'roc_auc' for classification else 'r2' for regression.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    def _score(y_true, y_hat):
        if metric == "roc_auc":
            if len(np.unique(y_true)) < 2:  # degenerate fold
                return np.nan
            return roc_auc_score(y_true, y_hat)
        return r2_score(y_true, y_hat)

    # baseline
    base_scores = []
    for tr, te in tscv.split(X):
        mdl = fit_fn(X.iloc[tr], y.iloc[tr])
        yhat = pred_fn(mdl, X.iloc[te])
        base_scores.append(_score(y.iloc[te], yhat))
    base = float(np.nanmean(base_scores))

    # permutations
    perm_scores = []
    rng = np.random.default_rng(0)
    for _ in range(n_permutations):
        perm = y.sample(frac=1.0, replace=False, random_state=int(rng.integers(1e9))).reset_index(drop=True)
        X_ = X.reset_index(drop=True)
        # same CV splits by index length
        fold_scores = []
        for tr, te in tscv.split(X_):
            mdl = fit_fn(X_.iloc[tr], perm.iloc[tr])
            yhat = pred_fn(mdl, X_.iloc[te])
            # Score against permuted labels in test
            fold_scores.append(_score(perm.iloc[te], yhat))
        perm_scores.append(np.nanmean(fold_scores))
    return {"baseline": base, "perm_mean": float(np.nanmean(perm_scores)), "perm_std": float(np.nanstd(perm_scores))}

def label_shuffle_sanity_check(
    X: pd.DataFrame, y: pd.Series,
    fit_fn: Callable[[pd.DataFrame, pd.Series], Any],
    pred_fn: Callable[[Any, pd.DataFrame], np.ndarray],
    n_splits: int = 5,
    metric: str = "roc_auc"
) -> float:
    """
    Single run of total label shuffle; expected performance ~0.5 AUC or ~0 R^2.
    """
    from sklearn.utils import shuffle
    y_shuf = pd.Series(shuffle(y.values, random_state=123), index=y.index)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    def _score(y_true, y_hat):
        return roc_auc_score(y_true, y_hat) if metric=="roc_auc" else r2_score(y_true, y_hat)

    scores = []
    for tr, te in tscv.split(X):
        m = fit_fn(X.iloc[tr], y_shuf.iloc[tr])
        yhat = pred_fn(m, X.iloc[te])
        scores.append(_score(y_shuf.iloc[te], yhat))
    return float(np.nanmean(scores))
