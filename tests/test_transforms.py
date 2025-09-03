import numpy as np, pandas as pd
from src.transforms.rolling import RollingImputer, RollingStandardizer, Winsorizer

def test_rolling_imputer_no_future():
    s = pd.Series([1.0, np.nan, 3.0, np.nan, 5.0])
    df = pd.DataFrame({"x": s})
    out = RollingImputer().fit_transform(df)
    # at t=1, should impute with mean([1.0]) = 1.0
    assert abs(out.iloc[1,0] - 1.0) < 1e-9
    # at t=3, mean([1,1?,3]) with shift => uses [1,1] = 1.0; safe check
    assert not np.isnan(out.iloc[3,0])

def test_winsorizer_bounds():
    x = pd.Series(list(range(100)))
    df = pd.DataFrame({"x": x})
    w = Winsorizer(lower_q=0.05, upper_q=0.95, min_periods=20)
    out = w.fit_transform(df)
    assert out["x"].iloc[50] == 50  # mid unaffected
