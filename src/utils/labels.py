# src/utils/labels.py
import numpy as np
import pandas as pd

def triple_barrier_labels(prices: pd.Series,
                          vol: pd.Series,
                          horizon: int,
                          up_mult: float = 1.0,
                          dn_mult: float = 1.0) -> pd.Series:
    """
    prices: close (or mid) indexed by timestamp
    vol:    estimated daily (or bar) volatility in *returns* (e.g., rolling std of log returns)
    horizon: max bars to hold (vertical barrier)
    Returns {-1, 0, 1} for short/flat/long labels.
    """
    assert prices.index.equals(vol.index)
    up_lvl = prices * (1 + up_mult * vol)
    dn_lvl = prices * (1 - dn_mult * vol)

    labels = pd.Series(index=prices.index, dtype="float64")

    idx = prices.index.to_list()
    for i, t0 in enumerate(idx):
        t1_idx = min(i + horizon, len(idx) - 1)
        window = prices.iloc[i:t1_idx + 1]
        if window.empty: 
            continue
        hit_up = (window >= up_lvl.iloc[i]).idxmax() if (window >= up_lvl.iloc[i]).any() else None
        hit_dn = (window <= dn_lvl.iloc[i]).idxmax() if (window <= dn_lvl.iloc[i]).any() else None

        # First barrier hit wins; else compare to start
        if hit_up and hit_dn:
            labels.loc[t0] = 1.0 if hit_up <= hit_dn else -1.0
        elif hit_up:
            labels.loc[t0] = 1.0
        elif hit_dn:
            labels.loc[t0] = -1.0
        else:
            labels.loc[t0] = np.sign(prices.iloc[t1_idx] - prices.iloc[i])
    return labels.astype(int).replace(0, np.nan).fillna(0)
