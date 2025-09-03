# src/transforms/rolling.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

EPS = 1e-12

@dataclass
class RollingImputer:
    """Expanding-mean imputer fitted only on past data."""
    min_periods: int = 1

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        means = X.expanding(self.min_periods).mean().shift(1)
        return X.where(~X.isna(), means)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # stateless by design: use expanding mean of the provided X (shifted)
        means = X.expanding(self.min_periods).mean().shift(1)
        return X.where(~X.isna(), means)

@dataclass
class RollingStandardizer:
    """Expanding z-score: (x - mean_{<=t-1}) / std_{<=t-1}."""
    min_periods: int = 20
    clip_sigma: float | None = None  # e.g., 4.0 to winsorize

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        mu = X.expanding(self.min_periods).mean().shift(1)
        sd = X.expanding(self.min_periods).std().shift(1).replace(0, np.nan)
        Z = (X - mu) / (sd + EPS)
        if self.clip_sigma:
            Z = Z.clip(lower=-self.clip_sigma, upper=self.clip_sigma)
        return Z

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(X)  # stateless for time series

@dataclass
class Winsorizer:
    """Rolling/walk-forward winsorization using expanding quantiles (shifted)."""
    lower_q: float = 0.01
    upper_q: float = 0.99
    min_periods: int = 200

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        lo = X.expanding(self.min_periods).quantile(self.lower_q).shift(1)
        hi = X.expanding(self.min_periods).quantile(self.upper_q).shift(1)
        return X.clip(lower=lo, upper=hi)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit_transform(X)
