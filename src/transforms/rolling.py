from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass

EPS = 1e-12

def _split_numeric(X: pd.DataFrame):
    num = X.select_dtypes(include=[np.number])
    other_cols = [c for c in X.columns if c not in num.columns]
    other = X[other_cols]
    return num, other

def _recombine_like(original: pd.DataFrame, num_out: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    # put columns back in the original order
    out = pd.concat([num_out, other], axis=1)
    return out[original.columns]

@dataclass
class RollingImputer:
    """Expanding-mean imputer fitted only on past data."""
    min_periods: int = 1

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        num, other = _split_numeric(X)
        means = num.expanding(self.min_periods).mean().shift(1)
        num_out = num.where(~num.isna(), means)
        return _recombine_like(X, num_out, other)

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit_transform(X, y=y)


@dataclass
class RollingStandardizer:
    """Expanding z-score: (x - mean_{<=t-1}) / std_{<=t-1} on numeric columns only."""
    min_periods: int = 20
    clip_sigma: float | None = None  # e.g., 4.0 to winsorize

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        num, other = _split_numeric(X)
        mu = num.expanding(self.min_periods).mean().shift(1)
        sd = num.expanding(self.min_periods).std().shift(1).replace(0, np.nan)
        Z = (num - mu) / (sd + EPS)
        if self.clip_sigma is not None:
            Z = Z.clip(lower=-self.clip_sigma, upper=self.clip_sigma)
        return _recombine_like(X, Z, other)

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit_transform(X, y=y)


@dataclass
class Winsorizer:
    """Rolling winsorization (expanding quantiles, shifted) on numeric columns only."""
    lower_q: float = 0.01
    upper_q: float = 0.99
    min_periods: int = 200

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        num, other = _split_numeric(X)
        if num.shape[1] == 0:
            return X  # nothing numeric to do
        lo = num.expanding(self.min_periods).quantile(self.lower_q).shift(1)
        hi = num.expanding(self.min_periods).quantile(self.upper_q).shift(1)
        num_out = num.clip(lower=lo, upper=hi)
        return _recombine_like(X, num_out, other)

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit_transform(X, y=y)
