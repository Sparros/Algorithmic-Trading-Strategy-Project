# src/execution/sizer.py
import numpy as np
import pandas as pd

def prob_to_weight(p: pd.Series, w_max: float = 0.5) -> pd.Series:
    """
    Map calibrated probability (for class 1 = up) to target weight in [-w_max, w_max].
    """
    raw = 2.0 * p - 1.0
    return raw.clip(-w_max, w_max)

def volatility_target(weights: pd.Series, returns: pd.Series, target_vol=0.10, lookback=63):
    vol = returns.rolling(lookback).std().shift(1) * np.sqrt(252.0)
    scale = (target_vol / vol).replace([np.inf, -np.inf], np.nan).clip(0, 10).fillna(0.)
    return (weights * scale).clip(-1.0, 1.0)

def apply_turnover_penalty(w_target: pd.Series, kappa: float = 0.001):
    """
    Penalize weight changes: w_t = w_tgt_t / (1 + kappa * |Δw|).
    """
    w = w_target.copy()
    prev = 0.0
    for t in w.index:
        dw = abs(w[t] - prev)
        w[t] = w[t] / (1.0 + kappa * dw)
        prev = w[t]
    return w
