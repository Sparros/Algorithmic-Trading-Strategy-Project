# src/execution/costs.py
import pandas as pd

def cost_model(trades: pd.Series, price: pd.Series,
               spread_bps: float = 5.0, fee_bps: float = 1.0, slippage_bps: float = 2.0) -> pd.Series:
    """
    trades: Δweight series (target - prev), same index as price.
    Returns per-period cost in return space.
    """
    bps = (spread_bps + fee_bps + slippage_bps) / 1e4
    return (trades.abs() * bps).reindex(price.index).fillna(0.0)
