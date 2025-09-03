# src/transforms/macro_asof.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

@dataclass
class AsOfMacroMerger:
    """
    Merge daily equities with macro series using 'asof' and series-specific publish lags.
    - macro_df: wide DataFrame with a Date index and macro columns.
    - publish_lag_days: dict[str,int] mapping column name -> extra business-day lag to avoid look-ahead.
    - tolerance_days: max distance for asof alignment.
    """
    publish_lag_days: dict[str, int]
    tolerance_days: int = 31

    def transform(self, stock_df: pd.DataFrame, macro_df: pd.DataFrame) -> pd.DataFrame:
        sdf = stock_df.copy().reset_index().rename(columns={stock_df.index.name or "index":"Date"})
        mdf = macro_df.copy().reset_index().rename(columns={macro_df.index.name or "index":"Date"})

        # apply lags per series
        for col, lag in (self.publish_lag_days or {}).items():
            if col in mdf.columns and lag and lag > 0:
                # shift by business days
                mdf[col] = mdf[col].shift(lag, freq="B")

        merged = pd.merge_asof(
            sdf.sort_values("Date"),
            mdf.sort_values("Date"),
            on="Date",
            direction="backward",
            tolerance=pd.Timedelta(days=self.tolerance_days)
        )
        merged = merged.set_index("Date").sort_index()
        return merged
